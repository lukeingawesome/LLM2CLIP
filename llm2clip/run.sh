MODEL=EVA02-CLIP-L-14-336
PRETRAINED=eva_clip
python -m torch.distributed.launch --nproc_per_node=4 \
	--use_env training/main.py \
        --enable-deepspeed \
        --grad-checkpointing \
        --name="final_llm2clip_caption" \
        --save-frequency 2  \
        --local-loss \
        --zeroshot-frequency 2 \
        --report-to="tensorboard, wandb" \
        --wandb-project-name="LLM2CLIP" \
        --wandb-notes="EVA02-CLIP-L-14-336" \
        --train-data "/data/csv/llm2clip/mimic_clip_train.csv" \
        --val-data "/data/csv/llm2clip/mimic_clip_test.csv" \
        --pretrained=${PRETRAINED} \
        --precision "fp16" \
        --warmup 0 \
        --batch-size=128 \
        --eval-batch-size=128 \
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
        --text-base="/model/llm2clip/llm2vec/8b_special/mntp/checkpoint-5779/" \
        --llm2vec-path="/model/llm2clip/llm2vec/8b_special/supervised/checkpoint-12535/" \
        --force-custom-clip \
        --optimizer="ap_adamw" \
        --zero-stage=1 \
        --dataset-type "cxr" \
        --csv-img-key "img_path" \
        --csv-caption-key "caption2_lite" \
        --rsna "/data/research/csv/rsna_test.csv" \
        --siim "/data/research/csv/siim_test.csv" \
        --openi "/data/csv/llm2clip/openi_clip_val.csv" \
        --chexpertplus "/data/csv/llm2clip/chexpert_clip_val.csv" \

