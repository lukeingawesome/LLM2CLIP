MODEL=EVA02-CLIP-L-14-336
PRETRAINED=eva_clip
python -m torch.distributed.launch --nproc_per_node=2 \
	--use_env training/main.py \
        --enable-deepspeed \
        --grad-checkpointing \
        --name="T_vitl336_mimic" \
        --save-frequency 2  \
        --zeroshot-frequency 2 \
        --report-to="tensorboard, wandb" \
        --wandb-project-name="LLM2CLIP" \
        --wandb-notes="EVA02-CLIP-L-14-336" \
        --train-data "/data/csv/llm2clip/mimic_clip.csv" \
        --pretrained=${PRETRAINED} \
        --precision "fp16" \
        --warmup 0 \
        --batch-size=150 \
        --eval-batch-size=150 \
        --log-every-n-steps 200 \
        --epochs=20 \
        --lr=1e-5 \
        --visual-lr=1e-5 \
        --text-lr=1e-5 \
        --wd=0.05 \
        --visual-wd=0.05 \
        --text-wd=0.05 \
        --ld=1.0 \
        --text-ld=1.01 \
        --visual-ld=0.85 \
        --grad-clip-norm=5.0 \
        --smoothing=0. \
        --workers=4 \
        --model=${MODEL} \
        --seed 4096 \
        --gather-with-grad \
        --local-loss \
        --text-base="meta-llama/Llama-3.2-3B" \
        --llm2vec-path="/data/research/tmp/checkpoint-12600/" \
        --force-custom-clip \
        --optimizer="ap_adamw" \
        --zero-stage=1 \
        --dataset-type "cxr" \
        --csv-img-key "img_path" \
        --csv-caption-key "caption2"
