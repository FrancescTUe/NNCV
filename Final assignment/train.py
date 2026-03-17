"""
This script implements a training loop for the model. It is designed to be flexible, 
allowing you to easily modify hyperparameters using a command-line argument parser.

### Key Features:
1. **Hyperparameter Tuning:** Adjust hyperparameters by parsing arguments from the `main.sh` script or directly 
   via the command line.
2. **Remote Execution Support:** Since this script runs on a server, training progress is not visible on the console. 
   To address this, we use the `wandb` library for logging and tracking progress and results.
3. **Encapsulation:** The training loop is encapsulated in a function, enabling it to be called from the main block. 
   This ensures proper execution when the script is run directly.

Feel free to customize the script as needed for your use case.
"""
import os
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
    InterpolationMode,
    RandomResizedCrop,
    RandomHorizontalFlip,
)
from ptflops import get_model_complexity_info

from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex
from torchvision import tv_tensors
from torchvision.transforms import v2
from model import Model, StudentModel


# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id

        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image


def get_args_parser():

    parser = ArgumentParser("Training script for a PyTorch HRNet model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="HRNet_v1-training", help="Experiment ID for Weights & Biases")

    return parser

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

WARMUP_STEPS = 1000 
def get_lr_sched(step, total_steps, base_lr):
    # Linear Warm-up
    if step < WARMUP_STEPS:
        return float(step) / float(max(1, WARMUP_STEPS))
    # Poly Decay after warm-up
    progress = (step - WARMUP_STEPS) / (total_steps - WARMUP_STEPS)
    return (1.0 - progress) ** 0.9

def distillation_loss(student_logits, teacher_logits, labels, T=2, alpha=0.5):
    # crossentropy
    soft_targets = F.softmax(teacher_logits/T, dim=1)
    log_probs = F.log_softmax(student_logits/T, dim=1)
    # KL divergence 
    kl_div = F.kl_div(log_probs, soft_targets, reduction='batchmean')*(T**2)
    # combine loss
    student_ce_loss = F.cross_entropy(student_logits, labels, ignore_index=255)
    
    return alpha*kl_div + (1-alpha)*student_ce_loss

def main(args):
    # Initialize wandb for logging
    wandb.init(
        project="5lsm0-cityscapes-segmentation",  # Project name in wandb
        name=args.experiment_id,  # Experiment name in wandb
        config=vars(args),  # Save hyperparameters
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducability
    # If you add other sources of randomness (NumPy, Random), 
    # make sure to set their seeds as well
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # we initialize more metrics
    f1_metric = MulticlassF1Score(num_classes=19, average='macro', ignore_index=255).to(device)
    miou_metric = MulticlassJaccardIndex(num_classes=19, ignore_index=255).to(device)

    # Define the transforms to apply to the data (training with data augmentation)
    img_transform = Compose([
    ToImage(),
    Resize((256, 512)),
    ToDtype(torch.float32, scale=True),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Target transform (mask)
    target_transform = Compose([
        ToImage(),
        Resize((256, 512), interpolation=InterpolationMode.NEAREST),
        ToDtype(torch.int64),  # no scaling
    ])

    # Load the dataset and make a split for training and validation
    train_dataset = Cityscapes(
    args.data_dir,
    split="train",
    mode="fine",
    target_type="semantic",
    transform=img_transform,
    target_transform=target_transform,
    )

    valid_dataset = Cityscapes(
    args.data_dir,
    split="val",
    mode="fine",
    target_type="semantic",
    transform=img_transform,
    target_transform=target_transform,
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )

    # Load teacher model
    teacher_model = Model(pretrained=False)
    state_dict = torch.load(
        'teacher_model.pt', 
        map_location=device,
        weights_only=True,
    )
    teacher_model.load_state_dict(
        state_dict, 
        strict=True,  # Ensure the state dict matches the model architecture
    )

    print(f"Teacher Parameters: {count_parameters(teacher_model):,}")

    teacher_model.eval().to(device)

    # Define the model
    student_model = StudentModel( 
        n_classes=19,  # 19 classes in the Cityscapes dataset
        ).to(device)
    print(f"Student Parameters: {count_parameters(student_model):,}")

    # Define the loss function (we add now class weights)
    cityscapes_weights = torch.tensor([
        2.81, 6.71, 3.78, 9.94, 9.77, 9.41, 10.27, 9.47, 2.88, 
        7.18, 3.85, 6.66, 9.59, 3.29, 9.55, 9.63, 9.63, 10.30, 9.55
    ], dtype=torch.float32)
    weights = cityscapes_weights.to(device)
    #criterion = nn.CrossEntropyLoss(weight=weights,ignore_index=255)  # Ignore the void class
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore the void class

    # Define the optimizer
    total_steps = len(train_dataloader) * args.epochs
    optimizer = AdamW(filter(lambda p: p.requires_grad, student_model.parameters()),
                       lr=args.lr, weight_decay=0.05)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: get_lr_sched(step, total_steps, args.lr))

    #Compute FLOPs 
    input_res = (3, 256, 512)

    with torch.cuda.device(0) if torch.cuda.is_available() else torch.cpu():
        t_macs, t_params = get_model_complexity_info(teacher_model, input_res, as_strings=True, print_per_layer_stat=False)
        s_macs, s_params = get_model_complexity_info(student_model, input_res, as_strings=True, print_per_layer_stat=False)

    print(f"Teacher Complexity: {t_macs} MACs")
    print(f"Student Complexity: {s_macs} MACs")

    # Training loop
    best_valid_loss = float('inf')
    current_best_model_path = None
    count_ep = 0 # counter for early stopping
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        # Training
        student_model.train()
        for i, (images, labels) in enumerate(train_dataloader):

            labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
            images, labels = images.to(device), labels.to(device)

            labels = labels.long().squeeze(1)  # Remove channel dimension
            with torch.no_grad():
                teacher_outputs = teacher_model(images)

            optimizer.zero_grad()
            outputs = student_model(images)
            loss = distillation_loss(outputs, teacher_outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step = epoch*len(train_dataloader)+i

            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
            }, step=global_step)
        wandb.log({"learning_rate": optimizer.param_groups[0]['lr']}, step=epoch)

        # Validation
        student_model.eval()
        f1_metric.reset()
        miou_metric.reset()
        with torch.no_grad():
            losses = []
            for i, (images, labels) in enumerate(valid_dataloader):

                labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                images, labels = images.to(device), labels.to(device)

                labels = labels.long().squeeze(1)  # Remove channel dimension

                outputs = student_model(images)
                loss = criterion(outputs, labels)
                losses.append(loss.item())

                predictions = outputs.softmax(1).argmax(1)
                f1_metric.update(predictions, labels)
                miou_metric.update(predictions, labels)

                if i == 0:
                    #predictions = outputs.softmax(1).argmax(1)

                    predictions = predictions.unsqueeze(1)
                    labels = labels.unsqueeze(1)

                    predictions = convert_train_id_to_color(predictions)
                    labels = convert_train_id_to_color(labels)

                    predictions_img = make_grid(predictions.cpu(), nrow=8)
                    labels_img = make_grid(labels.cpu(), nrow=8)

                    predictions_img = predictions_img.permute(1, 2, 0).numpy()
                    labels_img = labels_img.permute(1, 2, 0).numpy()

                    wandb.log({
                        "predictions": [wandb.Image(predictions_img)],
                        "labels": [wandb.Image(labels_img)],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)
            
            valid_loss = sum(losses) / len(losses)
            total_f1 = f1_metric.compute()
            total_miou = miou_metric.compute()

            wandb.log({
                "valid_loss": valid_loss,
                "val_f1": total_f1,
                "val_mIoU": total_miou,
            }, step=(epoch + 1) * len(train_dataloader) - 1)

            if valid_loss < best_valid_loss:
                count_ep = 0
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir, 
                    f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}.pt"
                )
                torch.save(model.state_dict(), current_best_model_path)

            else:
                count_ep+=1
        if count_ep == 10: # 10 epochs w/o doing anything
            print("Early stopping")
            break
    print("Training complete!")

    # Save the model
    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pt"
        )
    )
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
