# 🌱 Plant Disease Classification Using CNN

A PyTorch-based deep learning project for classifying plant leaf images into different disease categories using a Convolutional Neural Network (CNN).

## 📌 Project Overview

Plant diseases can significantly affect crop health and agricultural productivity. Early identification of diseases from leaf images can help support faster diagnosis.

This project builds an end-to-end image classification pipeline using **PyTorch and CNNs** to classify plant leaf images into six categories.

The project covers the complete workflow from dataset exploration and preprocessing to model training and evaluation.

## 🎯 Objectives

- Automatically classify plant leaf images
- Identify different plant disease categories
- Explore and visualize the image dataset
- Encode categorical labels into numerical values
- Split the dataset into training and validation sets
- Preprocess images for CNN training
- Implement a custom PyTorch Dataset
- Create PyTorch DataLoaders
- Train a CNN image classification model
- Track training and validation performance
- Save and load the best-performing model
- Evaluate the model using a confusion matrix

## 📊 Dataset

The dataset contains **1,500 plant leaf images** distributed equally across six classes.

### Classes

| Label | Class |
|---:|---|
| 0 | Early Blight |
| 1 | Healthy |
| 2 | Late Blight |
| 3 | Leaf Mold |
| 4 | Septoria Leaf Spot |
| 5 | Yellow Curl Virus |

Each class contains **250 images**.

## 🔄 Project Workflow

```text
Plant Leaf Images
        ↓
Dataset Loading
        ↓
Dataset Exploration
        ↓
DataFrame Creation
        ↓
Class Distribution Analysis
        ↓
Sample Image Visualization
        ↓
Label Encoding
        ↓
Train / Validation Split
        ↓
Image Preprocessing
        ↓
Custom PyTorch Dataset
        ↓
DataLoader
        ↓
CNN Model
        ↓
Model Training
        ↓
Best Model Selection
        ↓
Validation Evaluation
        ↓
Loss & Accuracy Analysis
        ↓
Confusion Matrix
```

## 🔍 Exploratory Data Analysis

The dataset was explored before model training to understand:

- Total number of images
- Number of images in each class
- Image paths
- Disease labels
- Class distribution
- Sample images

Sample images from all six classes were visualized to verify that the images were loaded correctly and matched their corresponding labels.

## 🏷️ Label Encoding

The original class names are converted into numerical labels so they can be used by the neural network.

```text
Early_Blight       → 0
Healthy            → 1
Late_Blight        → 2
Leaf_Mold          → 3
Septoria_Leaf_Spot → 4
Yellow_Curl_Virus  → 5
```

## ✂️ Train-Validation Split

The dataset was divided using an **80:20 stratified split**.

```text
Total Images      : 1500
Training Images   : 1200
Validation Images : 300
```

Stratification was used to maintain a similar class distribution in both training and validation datasets.

## 🖼️ Image Preprocessing

Before being passed to the CNN, each image is:

1. Loaded from its file path
2. Converted to RGB
3. Resized to `64 × 64`
4. Converted to a PyTorch tensor
5. Scaled to the required numerical range
6. Rearranged into PyTorch's channel-first format

The resulting image tensor has the shape:

```text
[3, 64, 64]
```

where:

- `3` = RGB channels
- `64` = image height
- `64` = image width

## 🧩 Custom PyTorch Dataset

A custom `PlantLeafDataset` class was implemented using `torch.utils.data.Dataset`.

The dataset class handles:

- Image path retrieval
- Label retrieval
- Image loading
- Image resizing
- RGB conversion
- Tensor conversion
- Label conversion

This makes the dataset compatible with PyTorch DataLoaders.

## 📦 DataLoader

PyTorch `DataLoader` is used to efficiently provide image batches during training and validation.

The training DataLoader uses:

```text
Batch Size = 32
Shuffle    = True
```

The validation DataLoader uses the validation dataset for model evaluation.

## 🤖 CNN Model

A Convolutional Neural Network is used to learn visual features from plant leaf images.

The model learns patterns such as:

- Edges
- Shapes
- Textures
- Spots
- Color patterns
- Disease-related visual features

The learned features are then passed through classification layers to predict one of the six disease classes.

## 🏋️ Model Training

The CNN is trained using PyTorch.

During training, the following metrics are recorded:

- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy

The model is evaluated after every epoch, and the model state with the best validation accuracy is saved.

The trained model is saved as:

```text
best_plant_disease_model.pth
```

## 📈 Training and Validation Loss

The loss curves are plotted to understand how the model's error changes during training.

The training loss generally decreases as the model learns useful features from the training images.

The validation loss is monitored to understand how well the model generalizes to unseen images.

## 📊 Training and Validation Accuracy

Training and validation accuracy are plotted across epochs.

These plots help compare:

- Model learning on the training dataset
- Model generalization on the validation dataset
- Changes in performance across epochs

## 💾 Best Model Selection

Instead of simply using the model from the final epoch, the project stores the model that achieves the highest validation accuracy.

```text
Best Validation Accuracy: 89.67%
```

The saved model is then loaded again for final validation evaluation.

## 🧪 Model Evaluation

The best-performing model is evaluated on the complete validation dataset.

The evaluation process:

1. Loads the saved model
2. Switches the model to evaluation mode
3. Generates predictions for validation images
4. Compares predictions with actual labels
5. Calculates validation accuracy
6. Generates the confusion matrix

## 📉 Confusion Matrix

A confusion matrix is used to analyze the classification performance of each disease class.

It shows:

- Correct predictions for each class
- Misclassified samples
- Which disease categories are confused with each other

This provides more detailed information than accuracy alone.

## 📈 Results

The CNN achieved a best validation accuracy of approximately:

```text
89.67%
```

The training and validation accuracy curves were used to monitor the learning process, while the confusion matrix was used to analyze class-level predictions.

## 🛠️ Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- PIL
- Google Colab

## 📁 Project Structure

```text
Plant-Disease-Classification-CNN/
│
├── Plant_Disease_Classification_CNN.ipynb
├── README.md
├── requirements.txt
├── .gitignore
```

<img width="704" height="470" alt="image" src="https://github.com/user-attachments/assets/9ffd9554-1435-464e-a502-32a6cac65fac" />

<img width="695" height="470" alt="image" src="https://github.com/user-attachments/assets/49d69cac-c300-498e-9816-0b9fbbf79757" />

<img width="1169" height="690" alt="image" src="https://github.com/user-attachments/assets/6861ac27-7cdf-4502-adf3-4e4a5898fb20" />


<img width="916" height="590" alt="image" src="https://github.com/user-attachments/assets/012d3d98-e9e8-441c-8a03-a19c787a96f5" />




## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Ebbeju-Lankapalli/Plant-Disease-Classification-CNN.git
```

### 2. Navigate to the Project

```bash
cd Plant-Disease-Classification-CNN
```

### 3. Create a Conda Environment

```bash
conda create -n plant-cnn python=3.12
conda activate plant-cnn
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Notebook

Open:

```text
Plant_Disease_Classification_CNN.ipynb
```

The notebook can be executed using:

- Google Colab
- Jupyter Notebook
- JupyterLab
- VS Code

## 💡 Key Learning Outcomes

Through this project, I practiced:

- Computer Vision fundamentals
- CNN-based image classification
- PyTorch Dataset implementation
- PyTorch DataLoader implementation
- Image preprocessing
- Label encoding
- Stratified train-validation splitting
- Neural network training
- Model checkpointing
- Loss and accuracy visualization
- Confusion matrix analysis
- Model evaluation

## 🔮 Future Improvements

Possible improvements include:

- Data augmentation
- Batch normalization
- Dropout
- Learning-rate scheduling
- Hyperparameter tuning
- Transfer learning
- ResNet-based classification
- MobileNet-based classification
- EfficientNet-based classification
- Larger and more diverse datasets
- Deployment using Streamlit or FastAPI
- Real-time plant disease prediction

## 👨‍💻 Author

**Ebbeju Lankapalli**

B.Tech Computer Science Engineering — AI/ML

GitHub:  
https://github.com/Ebbeju-Lankapalli

## ⭐ Repository

If you find this project useful for learning CNNs, PyTorch, or computer vision, consider giving the repository a star.

```text
Plant Disease Classification
        ↓
Computer Vision
        ↓
Convolutional Neural Network
        ↓
PyTorch
        ↓
Multi-Class Classification
        ↓
Model Evaluation
```
