wandb login


python3 train.py \
    --data-dir ./data/cityscapes \
    --ood-data-dir ./coco \
    --batch-size 128 \
    --epochs 60 \
    --lr 0.0002 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "OOD_v4_DUMMY" \
