# Histopathologic Cancer Detection Using Deep Learning 🧬

## 📚 Project Overview
This project explores deep learning approaches for cancer detection in histopathology images. It compares the performance of a **custom Convolutional Neural Network (CNN)** and a **pre-trained DenseNet-201** model to classify tissue samples as benign or malignant.

- **Dataset:** [PatchCamelyon (PCam)](https://github.com/basveeling/pcam)
- **Tech Stack:** Python, TensorFlow, Keras, Scikit-learn, Matplotlib
- **Techniques:** CNN from scratch, Transfer Learning with DenseNet, Data Augmentation, Evaluation Metrics

---

## 🖼 Sample Dataset Images

<div align="center">
  <img src="images/e9b14923-1549-4629-9def-2c0ebb10c620.png" width="800"/>
</div>

The PatchCamelyon (PCam) dataset contains high-resolution histopathology patches labeled as **benign** or **malignant**, helping automate metastasis detection.

---

## 🚀 Problem Statement
Manual examination of histopathology slides is labor-intensive and error-prone. This project aims to automate cancer diagnosis from tissue images using CNNs, thereby improving diagnostic accuracy, reducing human error, and supporting medical decision-making.

---

## 🏗 Project Structure
| File/Folder | Description |
|:---|:---|
| `Cancer_Detection_CNN_DenseNet.ipynb` | End-to-end notebook (preprocessing, modeling, evaluation) |
| `Project_Report.pdf` | Detailed project paper (methodology, results, discussion) |
| `/images/` | Dataset samples, confusion matrices, graphs (optional) |
| `requirements.txt` | List of Python package dependencies |
| `README.md` | Project summary and setup guide |

---

## 📈 Key Highlights
- Built a custom CNN achieving **91.9% validation accuracy**.
- Fine-tuned DenseNet-201 achieving **97.3% validation accuracy** using transfer learning.
- Applied **data augmentation** (rotation, flipping, zooming) for improved generalization.
- Evaluated models using **accuracy**, **precision**, **recall**, and **confusion matrices**.
- Analyzed limitations of custom CNN vs. pre-trained networks in complex datasets.

---

## 📊 Results Summary

| Model | Validation Accuracy |
|:---|:---|
| Custom CNN | 91.9% |
| DenseNet-201 (Transfer Learning) | 97.3% |

---

## 🔮 Future Work
- Apply **Grad-CAM** for model interpretability.
- Address dataset **class imbalance** with oversampling and weighted loss functions.
- Explore **ensemble methods** combining multiple deep learning models.

---

## 🛠 Setup Instructions

### Clone the Repository:
```bash
git clone https://github.com/your-username/histopathologic-cancer-detection.git
```
---

### Install Dependencies:
```bash
pip install -r requirements.txt
```

Tested on:
 - Python 3.9+
 - TensorFlow 2.8.0+
 - Scikit-learn 1.0.2+
 - Matplotlib 3.5+

---

## Run the Notebook:
Open Cancer_Detection_CNN_DenseNet.ipynb in Jupyter or Colab and execute all cells sequentially.

---

## 📜 Acknowledgments
Dataset sourced from PatchCamelyon Dataset
