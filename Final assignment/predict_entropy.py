"""
This script provides and example implementation of a prediction pipeline 
for a PyTorch U-Net model. It loads a pre-trained model, processes input 
images, and saves the predicted segmentation masks. 

You can use this file for submissions to the Challenge server. Customize 
the `preprocess` and `postprocess` functions to fit your model's input 
and output requirements.
"""
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Cityscapes
import os
from torchvision.transforms.v2 import (
    Compose, 
    ToImage, 
    Resize, 
    ToDtype, 
    Normalize,
    InterpolationMode,
)

from model import Model

# Fixed paths inside participant container
# Do NOT chnage the paths, these are fixed locations where the server will 
# provide input data and expect output data.
# Only for local testing, you can change these paths to point to your local data and output folders.
IMAGE_DIR = "./data/cityscapes"
OUTPUT_DIR = "/output"
MODEL_PATH = "model.pt"
COCO_DIR = "./coco"

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

def compute_batch_entropy(pred: torch.Tensor) -> torch.Tensor:
    """
    Computes the mean entropy for each image in a batch.
    Returns a tensor of shape (batch_size,)
    """
    pred_soft = torch.softmax(pred, dim=1)
    # entropy per pixel: shape (B, H, W)
    entropy_map = -torch.sum(pred_soft * torch.log(pred_soft + 1e-7), dim=1)
    # mean entropy per image: shape (B,)
    return torch.mean(entropy_map, dim=(1, 2))

def postprocess(pred: torch.Tensor, original_shape: tuple) -> np.ndarray:
    # Implement your postprocessing steps here
    # For example, resizing back to original shape, converting to color mask, etc.
    # Return a numpy array suitable for saving as an image
    pred_soft = nn.Softmax(dim=1)(pred)
    pred_max = torch.argmax(pred_soft, dim=1, keepdim=True)  # Get the class with the highest probability
    prediction = Resize(size=original_shape, interpolation=InterpolationMode.NEAREST)(pred_max)

    prediction_numpy = prediction.cpu().detach().numpy()
    prediction_numpy = prediction_numpy.squeeze()  # Remove batch and channel dimensions if necessary

    return prediction_numpy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")

    entropy_stats = {
            "cityscapes_val": [],
            "coco_ood": []
        }

    # Load model
    model = Model(pretrained=False)
    state_dict = torch.load(
        MODEL_PATH, 
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(
        state_dict, 
        strict=True,  # Ensure the state dict matches the model architecture
    )
    model.eval().to(device)

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
        Resize((224, 448), interpolation=InterpolationMode.NEAREST),
        ToDtype(torch.int64),  # no scaling
    ])

    valid_dataset = Cityscapes(
    IMAGE_DIR,
    split="val",
    mode="fine",
    target_type="semantic",
    transform=img_transform,
    target_transform=target_transform,
    )

    # COCO Validation (Far-OOD)
    ood_valid_dataset = ImageDataset(
        root=os.path.join(COCO_DIR, "val2017"),
        transform=img_transform
    )

    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=64, 
        shuffle=False,
        num_workers=8
    )

    ood_valid_dataloader = DataLoader(
        ood_valid_dataset, 
        batch_size=64, 
        shuffle=False,
        num_workers=8
    )

    with torch.no_grad():
        # we evaluate the cityscapes dataset
        for images, _ in enumerate(valid_dataloader):
            images = images.to(device)
            outputs = model(images)

            # Compute entropy for the whole batch
            batch_entropies = compute_batch_entropy(outputs)
            entropy_stats["cityscapes_val"].extend(batch_entropies.cpu().tolist())
                                                   
        for images, _ in ood_valid_dataloader:
            images = images.to(device)
            outputs = model(images)

            batch_entropies = compute_batch_entropy(outputs)
            entropy_stats["coco_ood"].extend(batch_entropies.cpu().tolist())

    np.savez(
        Path(OUTPUT_DIR) / "entropy_data.npz", 
        cityscapes=entropy_stats["cityscapes_val"], 
        coco=entropy_stats["coco_ood"]
    )

if __name__ == "__main__":
    main()