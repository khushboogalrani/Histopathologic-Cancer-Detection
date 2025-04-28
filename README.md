Histopathologic Cancer Detection Using Deep Learning
📚 Project Overview
This project explores the application of deep learning models for automated detection of cancer in histopathology images. It compares the performance of a custom-built Convolutional Neural Network (CNN) with a pre-trained DenseNet-201 model, evaluating their ability to classify tissue samples as benign or malignant.

Dataset: PatchCamelyon (PCam) Dataset

Models Used: Custom CNN, DenseNet-201 (transfer learning)

Tools & Libraries: Python, TensorFlow, Keras, Numpy, Matplotlib, Scikit-learn

🔍 Problem Statement
Accurate identification of metastases in lymph node histopathology images is crucial for effective cancer diagnosis and treatment planning. Manual examination is time-intensive and prone to human error. This project aims to automate cancer detection using deep learning to assist pathologists and improve diagnostic accuracy.

🏗 Project Structure
Final_Code.ipynb – Jupyter Notebook with data preprocessing, model training, evaluation.

Project_Paper.pdf – Detailed project report with background, methodology, results, and future work.

/images – (Optional) Visualizations like confusion matrices, accuracy curves.

🚀 Key Highlights
Developed a custom CNN achieving 91.9% validation accuracy.

Fine-tuned DenseNet-201 model using transfer learning achieving 97.3% validation accuracy.

Applied data augmentation techniques (rotation, flipping, zoom) to improve model generalization.

Evaluated models using metrics like Accuracy, Precision, Recall, Confusion Matrix.

📊 Results

Model	Validation Accuracy
Custom CNN	91.9%
DenseNet-201	97.3%
DenseNet-201 outperformed the custom CNN, demonstrating the effectiveness of transfer learning in medical image classification.

🔮 Future Work
Implement Grad-CAM to visualize model interpretability.

Address class imbalance with oversampling or adjusting class weights.

Explore ensemble methods combining outputs from CNN and DenseNet.

🛠 How to Run Locally
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/histopathologic-cancer-detection.git
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Open Final_Code.ipynb in Jupyter Notebook.

Run all cells sequentially.

📜 Acknowledgments
Dataset sourced from PCam Dataset (PatchCamelyon).
