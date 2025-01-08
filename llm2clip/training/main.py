import logging
import os
import sys
import random
from datetime import datetime
sys.path.append(os.getcwd())

import numpy as np
import torch
from torch.cuda.amp import GradScaler
import torch.nn as nn

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

from eva_clip import create_model_and_transforms, create_model_from_pretrained, trace_model, get_tokenizer

from training.data import get_data
from training.distributed import is_master, init_distributed_device, world_info_from_env, create_deepspeed_config
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import warmup_cosine_lr
from training.train import train_one_epoch, evaluate, extract_features
from training.optim import create_optimizer, get_all_parameters
from llm2vec_wrapper import LLM2VecWrapper as LLM2Vec
from peft import LoraConfig, get_peft_model

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def main(args):
    
    args, ds_init = parse_args(args)

    if ds_init is not None:
        create_deepspeed_config(args)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True # cudnn error 
        
    # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
    args.model = args.model.replace('/', '-')

    # get the name of the experiments
    if args.name is None:
        args.name = '-'.join([
            datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
            f"model_{args.model}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])
    else:
        args.name = '-'.join([
            args.name,
            datetime.now().strftime("%Y_%m_%d-%H")
        ])

    # discover initial world args early so we can log properly
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    args.log_path = None
    if is_master(args, local=args.log_local):
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        # if os.path.exists(args.log_path):
        #     print(
        #         "Error. Experiment already exists. Use --name {} to specify a new experiment."
        #     )
        #     return -1

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    if is_master(args):
        args.tensorboard_path = os.path.join(args.logs, args.name, "tensorboard") if args.tensorboard else ''
        args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''
        args.checkpoint_path = ''

    if args.copy_codebase:
        copy_codebase(args)

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    random_seed(args.seed, 0)
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_clip=args.force_custom_clip,
        force_patch_dropout=args.force_patch_dropout,
        pretrained_image=args.pretrained_image,
        pretrained_text=args.pretrained_text,
        pretrained_visual_model=args.pretrained_visual_model,
        pretrained_text_model=args.pretrained_text_model,
        image_mean=args.image_mean,
        image_std=args.image_std,
        cache_dir=args.cache_dir,
        skip_list=args.skip_list,
    )

    random_seed(args.seed, args.rank)
    if args.llm2vec_path:   
        print("Using LLM2Vec")
        text_model = LLM2Vec.from_pretrained(
            base_model_name_or_path=args.text_base,
            enable_bidirectional=True,
            peft_model_name_or_path=args.llm2vec_path,
            merge_peft=True,
            pooling_mode="mean",
            max_length=512,
            torch_dtype=torch.bfloat16,
        )
        # Add a trainable projection layer
        projection_layer = nn.Sequential(
            nn.LayerNorm(text_model.config.hidden_size),
            nn.Linear(text_model.config.hidden_size, model.visual.head.out_features)
        ).to(device)

        # Create a wrapper that combines LLM2Vec with projection
        class LLM2VecWithProjection(nn.Module):
            def __init__(self, llm2vec_model, projection):
                super().__init__()
                self.model = llm2vec_model
                self.projection = projection
                # self.tokenizer = llm2vec_model.tokenizer

            def forward(self, text):
                embeddings = self.model(text)
                return self.projection(embeddings)

            def lock(self, unlocked_layers=0, freeze_layer_norm=True):
                # Freeze LLM2Vec weights but keep projection trainable
                for param in self.model.parameters():
                    param.requires_grad = False

            def set_grad_checkpointing(self, enable=True):
                self.model.gradient_checkpointing_enable() if enable else self.model.gradient_checkpointing_disable()

        # Replace the text model with our wrapped version
        model.text = LLM2VecWithProjection(text_model.model, projection_layer)
        for param in model.parameters():
            # Check if parameter dtype is  Float (float32)
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.float16)
    total_n_parameters = sum(p.numel() for p in model.parameters())
    logging.info(f'number of total params: {total_n_parameters}')

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'number of params with requires_grad: {n_parameters}')

    if hasattr(model, 'visual'):
        # lora 
        if args.lora:
            config = LoraConfig(
                r=32,
                lora_alpha=32,
                target_modules=["q_proj",'v_proj','k_proj'],
                lora_dropout=0.1,
                bias="none",
            )
            model.visual = get_peft_model(model.visual, config)

        total_visual_n_parameters = sum(p.numel() for p in model.visual.parameters())
        logging.info(f'number of visual params: {total_visual_n_parameters}')
    if hasattr(model, 'text'):
        total_text_n_parameters = sum(p.numel() for p in model.text.parameters())
        logging.info(f'number of text params: {total_text_n_parameters}')

    model.to(device)
    model_without_ddp = model

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        logging.info("Lock image tower...")
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)

    if args.grad_checkpointing:
        if args.llm2vec_path:
            model.visual.set_grad_checkpointing()
        else:
            if args.lock_text:
                logging.info("Lock text tower...")  
                model.lock_text_tower(
                    unlocked_layers=args.lock_text_unlocked_layers,
                    freeze_layer_norm=args.lock_text_freeze_layer_norm)
            model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    # if args.distributed and not args.horovod:
    if args.distributed:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if not args.enable_deepspeed:
            ddp_args = {}
            if args.ddp_static_graph:
                # this doesn't exist in older PyTorch, arg only added if enabled
                ddp_args['static_graph'] = True
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
            model_without_ddp = model.module
            

    # create optimizer and scaler
    optimizer = None
    scaler = None
    if args.train_data or args.train_data_list or args.train_data_file or args.dataset_type == "synthetic":
        assert not args.trace, 'Cannot train with traced model'
                
        if not args.enable_deepspeed:
            scaler = GradScaler() if args.precision == "amp" else None
            optimizer = create_optimizer(args, model_without_ddp)
        else:
            scaler = None

            if args.optimizer != "lamb" and args.optimizer != "adamw":
                optimizer, optimizer_params = create_optimizer(
                    args,
                    model_without_ddp,
                    return_params=True)
                model, optimizer, _, _ = ds_init(
                    args=args,
                    model=model,
                    optimizer=optimizer,
                    model_parameters=optimizer_params,
                    dist_init_required=not args.distributed,
                )
            else:
                optimizer_params = get_all_parameters(args, model)
                model, optimizer, _, _ = ds_init(
                    args=args,
                    model=model,
                    model_parameters=optimizer_params,
                    dist_init_required=not args.distributed,
                )
        if is_master(args, local=args.log_local):
            logging.info(f"num of optimizer.param_groups: {len(optimizer.param_groups)}")

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        if args.enable_deepspeed:
            if os.path.exists(args.resume):
                import glob
                all_checkpoints = glob.glob(os.path.join(args.resume, 'epoch_*'))
                latest_ckpt = -1
                for ckpt in all_checkpoints:
                    t = ckpt.split('/')[-1].split('_')[1]
                    if t.isdigit():
                        latest_ckpt = max(int(t), latest_ckpt)
                if latest_ckpt >= 0:
                    start_epoch = latest_ckpt
                    _, client_states = model.load_checkpoint(args.resume, tag='epoch_%d' % latest_ckpt) #tag=f"epoch_{completed_epoch}"
                    logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {latest_ckpt})")
                else:
                    logging.info("=> no checkpoint found at '{}'".format(args.resume))
            else:
                logging.info("=> '{}' is not existing!".format(args.resume))
        else:
            if os.path.isfile(args.resume):
                checkpoint = torch.load(args.resume, map_location='cpu')
                if 'epoch' in checkpoint:
                    # resuming a train checkpoint w/ epoch and optimizer state
                    start_epoch = checkpoint["epoch"]
                    sd = checkpoint["state_dict"]
                    if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                        sd = {k[len('module.'):]: v for k, v in sd.items()}
                    model.load_state_dict(sd)
                    if optimizer is not None:
                        optimizer.load_state_dict(checkpoint["optimizer"])
                    if scaler is not None and 'scaler' in checkpoint:
                        scaler.load_state_dict(checkpoint['scaler'])
                    logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
                else:
                    # loading a bare (model only) checkpoint for fine-tune or evaluation
                    model.load_state_dict(checkpoint)
                    logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
            else:
                logging.info("=> no checkpoint found at '{}'".format(args.resume))

    # initialize datasets
    data = get_data(args, (preprocess_train, preprocess_val), epoch=start_epoch, tokenizer=text_model.tokenizer if args.llm2vec_path else get_tokenizer(args.model))
    assert len(data), 'At least one train or eval dataset must be specified.'

    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = data["train"].dataloader.num_batches * args.epochs
        if is_master(args):
            logging.info(f"total_steps: {total_steps}")
        scheduler = warmup_cosine_lr(optimizer, args, total_steps)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            config=vars(args),
            settings=wandb.Settings(start_method="fork")
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    if args.extract_features:
        with torch.no_grad():
            extract_features(model, data, args, device)
        return
        
    if 'train' not in data:
        evaluate(model, tokenizer, data, start_epoch, args, writer)
        return

    # torch.cuda.synchronize()
    # evaluate(model, data, -1, args, writer)
    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')
        tokenizer = text_model.tokenizer if args.llm2vec_path else get_tokenizer(args.model)
        # text_config = text_model.config if args.llm2vec_path else None
        train_one_epoch(model, tokenizer, data, epoch, optimizer, scaler, scheduler, args, writer)
        completed_epoch = epoch + 1

        if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
            evaluate(model, tokenizer, data, completed_epoch, args, writer)

        # Saving checkpoints.
        # is_master(args) can not be here while using deepspped, otherwise ckpt can not be saved
        if args.logs and args.logs.lower() != 'none' and args.enable_deepspeed:
            deepspeed_checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
            if completed_epoch == args.epochs or (
                    args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
                ):
                    client_state = {'epoch': completed_epoch}
                    model.save_checkpoint(save_dir=deepspeed_checkpoint_path, tag="epoch_%s" % str(completed_epoch), client_state=client_state)

        elif args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            if args.save_most_recent:
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_latest.pt"),
                )

    if args.wandb and is_master(args):
        wandb.finish()


def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main(sys.argv[1:])
