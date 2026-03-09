wandb login


python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 32 \
    --epochs 50 \
    --lr 0.0005 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "Pretrained_model-v3" \
