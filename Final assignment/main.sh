wandb login


python3 train.py \
    --data-dir ./data/cityscapes \
    --ood-data-dir ./coco \
    --batch-size 128 \
    --epochs 100 \
    --lr 0.005 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "OOD_v7" \
