from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
import torch

def get_loaders(data_dir, batch_size=32):

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        normalize
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    train_data = datasets.ImageFolder(f"{data_dir}/train", transform=train_transform)
    val_data = datasets.ImageFolder(f"{data_dir}/val", transform=val_transform)

    # 🔥 FIX: Weighted Sampling (NO Subset)
    labels = [label for _, label in train_data.samples]
    class_counts = Counter(labels)

    weights = [1.0 / class_counts[label] for label in labels]
    weights = torch.DoubleTensor(weights)

    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    return train_loader, val_loader, len(train_data.classes)