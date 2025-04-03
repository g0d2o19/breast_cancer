# Breast Cancer Classification Project

## Overview
This project implements a machine learning solution for breast cancer diagnosis classification using traditional machine learning algorithms. The system analyzes various features from diagnostic data to predict whether breast masses are benign or malignant, with a focus on maximizing both accuracy and recall for the malignant class.

## Dataset
The project uses the Breast Cancer Wisconsin (Diagnostic) Dataset, which includes features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. Features describe characteristics of the cell nuclei present in the image.

Classes:
- 0: Benign
- 1: Malignant

The dataset has a slight class imbalance with a ratio of approximately 1.68:1 (benign:malignant).

## Project Structure
```
├── data.csv                  # Breast cancer dataset
├── main.py                   # Main script containing data processing and model training
└── README.md                 # Project documentation
```

## Methodology

### Data Preprocessing
1. Handling missing values:
   - For numerical features: Replaced with median values
   - For categorical features: Replaced with mode values
2. Feature scaling using StandardScaler
3. Exploratory data analysis with histograms and count plots
4. Label encoding for categorical variables
5. Train-test split (80-20) with stratification to maintain class distribution

### Models Evaluated
The project evaluates several machine learning algorithms:
- Logistic Regression (with different regularization techniques)
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM) with different kernels:
  - Linear
  - RBF (Radial Basis Function)
  - Polynomial
- Decision Tree (with analysis of depth and pruning effects)

### Hyperparameter Tuning
GridSearchCV with cross-validation was used to find optimal hyperparameters for logistic regression, focusing on regularization strength and type.

### Evaluation Metrics
Models were evaluated using:
- Accuracy
- Precision, Recall, and F1-score
- Confusion matrices
- ROC curves and AUC scores

## Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Logistic Regression | 97.37% | 0.9714 | 0.9714 | 0.9714 | 0.9967 |
| K-Nearest Neighbors | 95.61% | 0.9571 | 0.9429 | 0.9499 | 0.9821 |
| SVM (RBF Kernel) | 96.49% | 0.9643 | 0.9643 | 0.9643 | 0.9947 |
| Decision Tree | 91.23% | 0.9143 | 0.9143 | 0.9143 | 0.9216 |

### Key Findings

1. **Logistic Regression Analysis:**
   - Best overall performance (97.37% accuracy)
   - L2 regularization with C=0.1 provided optimal results
   - L1 regularization created sparse feature coefficients (feature selection)

2. **SVM Kernel Comparison:**
   - RBF kernel performed best (96.49% accuracy)
   - Linear kernel showed strong performance (96%)
   - Polynomial kernel underperformed (88% accuracy)

3. **Decision Tree Analysis:**
   - Default parameters: 92% accuracy
   - Pruned tree: 90% accuracy but better generalization
   - Key features identified through importance analysis

4. **K-Nearest Neighbors:**
   - 96% accuracy with K=5
   - Strong performance but slightly lower recall for the minority class

## Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Installation
```bash
git clone https://github.com/yourusername/breast-cancer-classification.git
cd breast-cancer-classification
pip install -r requirements.txt
```

## Usage
1. Place your dataset at the appropriate location
2. Update the path in the script if necessary
3. Run the main script:
```bash
python main.py
```

## Future Improvements
- Implement ensemble methods like Random Forest and Gradient Boosting
- Explore feature engineering and selection techniques
- Apply more advanced resampling techniques for handling class imbalance
- Implement deep learning models for comparison

## License
[Your License Here]

## Acknowledgements
This project utilizes the Breast Cancer Wisconsin (Diagnostic) Dataset for cancer classification.
