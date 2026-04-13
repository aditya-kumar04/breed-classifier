# Training script for breed classification model
import torch
import torch.nn as nn
import torchvision.models as models
from dataset import get_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, num_classes = get_loaders("data")

from torchvision.models import resnet50, ResNet50_Weights

model = resnet50(weights=ResNet50_Weights.DEFAULT)


# Replace final layer
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last layer block (layer4)
for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), "lr": 1e-5},
    {"params": model.fc.parameters(), "lr": 1e-4}
])

def train(model, loader):
    model.train()
    total_loss = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    correct = 0
    top3_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # Top-1
            correct += (preds == labels).sum().item()

            # Top-3
            top3 = torch.topk(outputs, 3, dim=1).indices
            for i in range(labels.size(0)):
                if labels[i] in top3[i]:
                    top3_correct += 1

            total += labels.size(0)

    return correct/total, top3_correct/total

epochs = 15
best_acc = 0

for epoch in range(epochs):
    loss = train(model, train_loader)
    acc, top3 = evaluate(model, val_loader)

    print(f"Epoch {epoch+1}")
    print(f"Loss: {loss:.4f}")
    print(f"Val Accuracy: {acc:.4f}")
    print(f"Top-3 Accuracy: {top3:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "models/best_model.pth")
        print("Model saved!")