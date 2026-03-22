wandb login


python3 train.py \
    --data-dir ./data/cityscapes \
    --ood-data-dir ./coco \
    --batch-size 64 \
    --epochs 60 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "OOD_v1" \
