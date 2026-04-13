import torch
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from torchvision.models import resnet50
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names (MUST match training folder order)
classes = [
    "Jaffrabadi", "Mehsana", "Murrah",
    "Nagori", "Red_Sindhi", "Sahiwal",
    "Surti", "Tharparkar", "Toda"
]

# Load model
def load_model():
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 9)

    model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
    model = model.to(device)
    model.eval()

    return model

model = load_model()

# Transform (MUST match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 🔥 Inference function
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        temperature = 2.0  # try between 1.5–3.0
        probs = torch.softmax(outputs / temperature, dim=1)

    confidence, pred = torch.max(probs, 1)
    top3_prob, top3_idx = torch.topk(probs, 3)

    return {
        "prediction": classes[pred.item()],
        "confidence": float(confidence.item()),
        "top3": [
            {
                "breed": classes[idx],
                "confidence": float(prob)
            }
            for prob, idx in zip(top3_prob[0], top3_idx[0])
        ]
    }


# 🧪 Batch Testing (Clean + Useful)
if __name__ == "__main__":

    folder = "data/test/Murrah"  # change class here
    actual_label = os.path.basename(folder)

    correct = 0
    total = 0

    print(f"\n📂 Testing Folder: {folder}")
    print(f"🎯 Actual Class: {actual_label}\n")

    for img_name in os.listdir(folder)[:10]:  # test 10 images
        image_path = os.path.join(folder, img_name)

        result = predict(image_path)
        predicted = result["prediction"]
        confidence = result["confidence"]

        is_correct = predicted == actual_label

        if is_correct:
            correct += 1
        total += 1

        print(f"Image: {img_name}")
        print(f"→ Predicted: {predicted} ({confidence:.4f})")
        print(f"→ Correct: {is_correct}")
        print(f"→ Top-3: {result['top3']}")
        print("-" * 50)

    print("\n📊 Summary:")
    print(f"Accuracy: {correct}/{total} = {correct/total:.2f}")