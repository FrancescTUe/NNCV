wandb login


python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.00005 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "Pretrained_model-v2" \
