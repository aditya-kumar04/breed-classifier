# 🐃 Buffalo Breed Identification System

An end-to-end AI-powered system for identifying buffalo breeds from images using deep learning and explainable AI techniques.

---

## 🚀 Overview

This project leverages a Convolutional Neural Network (CNN) based on ResNet50 to classify buffalo breeds from images. It includes a full pipeline from training to deployment, along with an interactive web interface and model interpretability using Grad-CAM.

---

## 🎯 Features

- 🧠 Deep learning model (ResNet50-based classifier)
- 📸 Image-based breed prediction
- 📊 Confidence scores + Top-3 predictions
- 🔍 Grad-CAM visualization (model explainability)
- 🌐 Interactive web app using Streamlit
- ⚡ Real-time inference
- 📦 Clean and modular inference pipeline

---

## 🐄 Supported Breeds

- Murrah
- Jaffrabadi
- Mehsana
- Nagori
- Red Sindhi
- Sahiwal
- Surti
- Tharparkar
- Toda

---

## 🧠 Model Architecture

- Backbone: ResNet50
- Transfer Learning:
  - Frozen base layers
  - Fine-tuned last block (`layer4`) and fully connected layer
- Loss Function: CrossEntropyLoss with label smoothing
- Optimizer: Adam (differential learning rates)

---

## 📂 Project Structure
Breed/
├── app.py # Streamlit web app
├── predict.py # Inference + batch testing
├── train.py # Training script
├── dataset.py # Data loader
├── models/
│ └── best_model.pth # Trained model weights
├── data/
│ ├── train/
│ └── test/
├── requirements.txt
└── README.md


---

## ⚙️ Installation

### 1. Clone repository
git clone https://github.com/aditya-kumar04/breed-classifier.git
cd breed-classifier


### 2. Create environment
conda create -n breed python=3.10
conda activate breed


### 3. Install dependencies
pip install -r requirements.txt


---

## 🧪 Run Inference (CLI)
python predict.py


---

## 🌐 Run Web App
streamlit run app.py


Then open in browser:
http://localhost:8501


---

## 🔍 Explainability (Grad-CAM)

The app includes Grad-CAM visualizations to highlight regions of the image influencing predictions.

This helps:
- Debug model behavior
- Detect bias
- Improve interpretability

---

## ⚠️ Observations & Limitations

- Model shows confusion between visually similar breeds (e.g., Murrah vs Sahiwal)
- High confidence predictions may indicate overfitting
- Performance depends heavily on dataset quality and diversity

---

## 🚀 Future Improvements

- Improve dataset balance and quality
- Add attention mechanisms (Vision Transformers / CBAM)
- Multi-image batch prediction
- Mobile deployment
- Real-time video inference

---

## 📊 Results

- Validation Accuracy: ~92%
- Top-3 Accuracy: High consistency across classes
- Identified model bias using Grad-CAM

---

## 🧑‍💻 Tech Stack

- Python
- PyTorch
- Torchvision
- Streamlit
- OpenCV
- NumPy

---

## 📌 Use Cases

- Livestock identification
- Dairy farm management
- Veterinary support tools
- Agricultural AI applications

---

## 🤝 Contributing

Contributions are welcome. Feel free to open issues or submit pull requests.

---

## 📜 License

MIT License

---

## 👤 Author

Aditya Kumar Jha  