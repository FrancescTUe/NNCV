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
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
    InterpolationMode,
)
from ptflops import get_model_complexity_info

from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex
from model import Model, FM_OODModel


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

class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        # we list all image files
        self.images = [f for f in os.listdir(root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.images[index])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0  
    
def get_args_parser():

    parser = ArgumentParser("Training script for a PyTorch HRNet model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--ood-data-dir", type=str, default="./coco", help="Path to COCO for OOD validation")
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

def flow_matching_loss(flow_head, x1):
    loss_fn = nn.MSELoss()
    # x0 is normal noise 
    x0 = torch.randn_like(x1)
    
    # we sample a random time t between 0 and 1
    t = torch.rand(x1.shape[0], 1, device=x1.device)
    
    # linear interpolation: x_t = t*x1 + (1-t)*x0
    xt = t*x1 + (1-t)*x0
    
    # we compute the target velocity
    target_velocity = x1 - x0
    
    # we predict the velocity
    predicted_velocity = flow_head(t, xt)
    
    # MSE Loss between velocities
    return loss_fn(predicted_velocity, target_velocity)

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

    # Define the transforms to apply to the data (training with data augmentation)
    img_transform = Compose([
    ToImage(),
    Resize((512, 1024)),
    ToDtype(torch.float32, scale=True),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Target transform (mask)
    target_transform = Compose([
        ToImage(),
        Resize((224, 448), interpolation=InterpolationMode.NEAREST),
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

    # COCO Validation (Far-OOD)
    ood_valid_dataset = ImageDataset(
        root=os.path.join(args.ood_data_dir, "val2017"),
        transform=img_transform
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

    ood_valid_dataloader = DataLoader(
        ood_valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )

    ood_model = FM_OODModel().to(device)

    # Define the optimizer
    optimizer = AdamW(filter(lambda p: p.requires_grad, ood_model.flow_head.parameters()),
                       lr=args.lr, weight_decay=0.05)

    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: get_lr_sched(step, total_steps, args.lr))

    # Training loop
    best_separation_ratio = 0
    current_best_model_path = None
    count_ep = 0 # counter for early stopping
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")
        # Training
        ood_model.train()
        for i, (images, labels) in enumerate(train_dataloader):
            labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
            images, labels = images.to(device), labels.to(device)

            outputs = ood_model.encoder(images, output_hidden_states=True) 

            #s2 = torch.mean(outputs.hidden_states[2], dim=[2, 3])
            #s3 = torch.mean(outputs.hidden_states[3], dim=[2, 3])
            #s4 = torch.mean(outputs.hidden_states[4], dim=[2, 3])
            #multi_scale_latent = torch.cat([s2, s3, s4], dim=1)

            features = outputs.last_hidden_state

            latent_vector = torch.mean(features, dim=[2,3])
            #latent = F.normalize(latent_vector, p=2, dim=1)

            optimizer.zero_grad()
            loss = flow_matching_loss(ood_model.flow_head, features)
            loss.backward()
            optimizer.step()
            #scheduler.step()

            global_step = epoch*len(train_dataloader)+i

            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
            }, step=global_step)
        wandb.log({"learning_rate": optimizer.param_groups[0]['lr']}, step=epoch)

        # Validation
        ood_model.eval()
        cityscapes_scores = []
        coco_scores = []
        with torch.no_grad():
            # we evaluate the cityscapes dataset
            for images, _ in valid_dataloader:
                ood_score = ood_model(images.to(device))
                cityscapes_scores.extend(ood_score.cpu().tolist())

            #we evaluate the COCO dataset
            #for images, _ in ood_valid_dataloader:
             #   ood_score = ood_model(images.to(device))
              #  coco_scores.extend(ood_score.cpu().tolist())
            for _ in range(len(ood_valid_dataloader)):
                # Generate random noise with the same shape as the SegFormer features
                noise_latent = torch.randn((args.batch_size, 256), device=device)
                noise_latent = F.normalize(noise_latent, p=2, dim=1)
                ood_score = ood_model.compute_log_likelihood(noise_latent)
                coco_scores.extend(ood_score.cpu().tolist())

            separation_ratio = (sum(coco_scores) / len(coco_scores)) / (sum(cityscapes_scores) / len(cityscapes_scores))
            wandb.log({
                "avg_id_score": sum(cityscapes_scores) / len(cityscapes_scores),
                "avg_ood_score": sum(coco_scores) / len(coco_scores),
                "separation_ratio": separation_ratio
            }, step=(epoch + 1) * len(train_dataloader) - 1)

            if separation_ratio > best_separation_ratio:
                count_ep = 0
                best_separation_ratio = separation_ratio
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir, 
                    f"best_model-epoch={epoch:04}-sep_rat={separation_ratio:04}.pt"
                )
                torch.save(ood_model.flow_head.state_dict(), current_best_model_path)

            else:
                count_ep+=1
        if count_ep == 10: # 10 epochs w/o doing anything
            print("Early stopping")
            break
    print("Training complete!")

    # Save the model
    torch.save(
        ood_model.flow_head.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-sep_rat={separation_ratio:04}.pt"
        )
    )
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
