import os
from argparse import ArgumentParser
import numpy as np
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

from torchmetrics.classification import MulticlassF1Score
from model import Model, FM_OODModel

IMAGE_DIR = "./data/cityscapes"
OUTPUT_DIR = "./output"
COCO_DIR = "./coco"
MODEL_PATH = "ood_model.pt"
SEG_MODEL_PATH = "teacher_model.pt"

id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

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
    
def main(num_batches=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_transform = Compose([
    ToImage(),
    Resize((512, 1024)),
    ToDtype(torch.float32, scale=True),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Target transform (mask)
    target_transform = Compose([
        ToImage(),
        Resize((256, 512), interpolation=InterpolationMode.NEAREST),
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

    ood_model = FM_OODModel().to(device)

    id_latents = []
    ood_latents = []
    ood_model.eval()

    with torch.no_grad():
        print("Extracting ID latents...")
        for i, (images, _) in enumerate(valid_dataloader):
            if i >= num_batches: break
            outputs = ood_model.encoder(images.to(device))
            # GAP as per your model.py
            latent = torch.mean(outputs.last_hidden_state, dim=[2, 3])
            latent = F.normalize(latent, p=2, dim=1)
            id_latents.append(latent.cpu().numpy())

        print("Extracting OOD latents...")
        for i, (images, _) in enumerate(ood_valid_dataloader):
            if i >= num_batches: break
            outputs = ood_model.encoder(images.to(device))
            latent = torch.mean(outputs.last_hidden_state, dim=[2, 3])
            latent = F.normalize(latent, p=2, dim=1)
            ood_latents.append(latent.cpu().numpy())

    np.save(os.path.join(OUTPUT_DIR,"id_latents.npy"), np.concatenate(id_latents, axis=0))
    np.save(os.path.join(OUTPUT_DIR,"ood_latents.npy"), np.concatenate(ood_latents, axis=0))
    print("Done! Files saved.")

def main_2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_transform = Compose([
    ToImage(),
    Resize((512, 1024)),
    ToDtype(torch.float32, scale=True),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_transform_seg = Compose([
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


    valid_dataset_seg = Cityscapes(
    IMAGE_DIR,
    split="val",
    mode="fine",
    target_type="semantic",
    transform=img_transform_seg,
    target_transform=target_transform,
    )

    valid_dataloader_seg = DataLoader(
        valid_dataset_seg, 
        batch_size=1, 
        shuffle=False,
        num_workers=8
    )

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
        batch_size=1, 
        shuffle=False,
        num_workers=8
    )

    ood_valid_dataloader = DataLoader(
        ood_valid_dataset, 
        batch_size=1, 
        shuffle=False,
        num_workers=8
    )

    model = Model(pretrained=False)
    state_dict = torch.load(
            SEG_MODEL_PATH, 
            map_location=device,
            weights_only=True,
        )
    model.load_state_dict(
        state_dict, 
        strict=True,  # Ensure the state dict matches the model architecture
    )
    model.eval().to(device)


    ood_model = FM_OODModel().to(device)

    state_dict = torch.load(
            MODEL_PATH, 
            map_location=device,
            weights_only=True,
        )
    ood_model.flow_head.load_state_dict(
        state_dict, 
        strict=True,  # Ensure the state dict matches the model architecture
    )
    ood_model.eval().to(device)

    cityscapes_scores = []
    coco_scores = []
    cityscapes_dice_scores = []
    ood_model.eval()

    dice_metric = MulticlassF1Score(num_classes=19, average='macro', ignore_index=255).to(device)
    with torch.no_grad():
        print("Saving scores for ID samples...")
        for i, (images, _) in enumerate(valid_dataloader):
            ood_score = ood_model(images.to(device))
            cityscapes_scores.extend(ood_score.cpu().tolist())

        print("Computing segmentation DICE for ID samples...")
        for i, (images, labels) in enumerate(valid_dataloader_seg):
            labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
            images, labels = images.to(device), labels.to(device)

            labels = labels.long().squeeze(1)  # Remove channel dimension

            outputs = model(images)

            predictions = outputs.softmax(1).argmax(1)
            batch_dice = dice_metric(predictions, labels)
            cityscapes_dice_scores.append(batch_dice.cpu().numpy())

        print("Saving scores for OOD samples...")
        for i, (images, _) in enumerate(ood_valid_dataloader):
            ood_score = ood_model(images.to(device))
            coco_scores.extend(ood_score.cpu().tolist())

    np.save(os.path.join(OUTPUT_DIR,"id_scores.npy"), np.array(cityscapes_scores))
    np.save(os.path.join(OUTPUT_DIR,"ood_scores.npy"), np.array(coco_scores))
    np.save(os.path.join(OUTPUT_DIR, "id_dice_scores.npy"), np.array(cityscapes_dice_scores))
    
    print("Done! Files saved.")


if __name__ == "__main__":
    main_2()
