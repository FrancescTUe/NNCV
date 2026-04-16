wandb login


python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 128 \
    --epochs 100 \
    --lr 0.005 \
    --num-workers 10 \
    --seed 42 \
    --ood_data_dir ./coco \
    --experiment-id "OOD_Model" \
