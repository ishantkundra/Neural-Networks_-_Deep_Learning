# 🤖 Signal Quality Classification and Digit Recognition using Deep Learning

## 📌 Project Overview

This project showcases the application of deep learning techniques in two challenging real-world domains:  
- **Signal Quality Classification** in Electronics & Telecommunication  
- **Digit Recognition** from natural street view images (SVHN dataset)  

Both parts involve data preprocessing, building and training neural network models using **TensorFlow**, and interpreting results through visualizations.

---

## 📡 Part A: Signal Quality Classification

### 🌐 Domain: Electronics & Telecommunication

A communications equipment manufacturing company wants to predict **signal quality** using measurable parameters from various signal tests.

### 📊 Dataset:
- `Part-1,2&3 - Signal.csv` – Contains sensor-based signal test features and signal quality labels

### 🧠 Objective:
Build a neural network classifier to determine the signal quality emitted by communication equipment.

### 🛠️ Key Steps:
- Loaded and explored the dataset
- Handled missing values and duplicate records
- Visualized class distribution and shared insights
- Split into training and testing datasets (70:30)
- Normalized features and transformed labels
- Designed, trained, and optimized neural network using **TensorFlow/Keras**
- Plotted:
  - Training vs validation loss
  - Training vs validation accuracy
- Improved performance by tuning architecture and comparing results

---

## 🏙️ Part B: Digit Recognition using SVHN Dataset

### 🛻 Domain: Autonomous Vehicles / Image Recognition

The **Street View House Numbers (SVHN)** dataset provides digit images from Google Street View. The goal is to classify the digit at the center of each image.

### 📦 Dataset:
- `Autonomous_Vehicles_SVHN_single_grey1.h5` – A cleaned HDF5 version of SVHN containing images centered around a single digit

### 🧠 Objective:
Build a **Convolutional Neural Network (CNN)** to recognize digits (0-9) from natural images.

### 🛠️ Key Steps:
- Loaded `.h5` file and explored data
- Split into `X_train`, `X_test`, `y_train`, `y_test`
- Reshaped, normalized, and one-hot encoded labels
- Visualized 10 sample images with labels
- Designed CNN architecture with Conv2D, MaxPooling, Flatten, Dense layers
- Trained the model with appropriate parameters
- Evaluated performance with classification metrics
- Plotted training & validation loss/accuracy curves and analyzed model behavior

---

## ⚙️ Tools, Libraries & Skills Used

- **Languages:** Python  
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, TensorFlow, Keras, h5py  
- **Skills:**
  - Deep Learning & Neural Networks
  - Convolutional Neural Networks (CNN)
  - Image preprocessing & normalization
  - Label encoding & one-hot transformation
  - Model evaluation with loss & accuracy plots
  - Electronics & signal-based data processing
  - TensorFlow/Keras pipeline for training

---

## 📁 Repository Structure

<pre>

.
├── code/
│   ├── Final_NN_IK.ipynb                         # Jupyter Notebook (both parts)
│   └── Final_NN_IK.html                          # Exported HTML report
│
├── dataset/
│   ├── Part-1,2&3 - Signal.csv                   # Signal quality dataset (Part A)
│   └── Autonomous_Vehicles_SVHN_single_grey1.h5  # SVHN digit recognition dataset (Part B)
│
├── Problem Statement/
│   └── NN - Problem - Statement.pdf              # Detailed task instructions
│
├── .gitignore
└── README.md                                     # This file

</pre>

---

## 💡 Key Learnings

- How to process both tabular and image data for neural networks  
- Designing, training, and tuning fully connected and CNN architectures  
- Working with real-world datasets like SVHN (image-based) and signal data (numeric)  
- Visualizing training progress and evaluating deep learning model effectiveness  
- Applying deep learning in telecom and computer vision domains

---

## ✍️ Author

**Ishant Kundra**  
📧 [ishantkundra9@gmail.com]
🎓 Master’s in Computer Science | AIML Track
