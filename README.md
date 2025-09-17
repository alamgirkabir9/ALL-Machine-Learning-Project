# Machine Learning ALL Project Repository

A comprehensive collection of machine learning projects implementing various prediction models for real-world applications.

## 📁 Project Structure

```
MachineLearning-ALL-Project/
├── DiabetesPredictFunction.ipynb    # Diabetes prediction model
├── student_performance.ipynb        # Student performance analysis
├── titanic_survival.ipynb          # Titanic survival prediction
└── README.md                       # Project documentation
```

## 🚀 Projects Overview

### 1. Diabetes Prediction Model
**File:** `DiabetesPredictFunction.ipynb`

Predicts the likelihood of diabetes based on medical indicators using machine learning classification algorithms.

**Features:**
- **Input Variables (8 features):**
  - Pregnancies
  - Glucose level
  - Blood Pressure
  - Skin Thickness
  - Insulin level
  - BMI (Body Mass Index)
  - Diabetes Pedigree Function
  - Age

- **Data Preprocessing:**
  - KNN Imputation for missing values (k=5 neighbors)
  - Feature scaling and normalization

- **Models Used:**
  - Logistic Regression
  - Random Forest Classifier

- **Output:** Binary classification (Diabetic/Not Diabetic)

### 2. Student Performance Analysis
**File:** `student_performance.ipynb`

Analyzes and predicts student academic performance using linear regression.

**Features:**
- **Input Variables:**
  - Hours Studied
  - Previous Scores
  - Extracurricular Activities
  - Sleep Hours
  - Sample Question Papers Practiced

- **Model Used:**
  - Linear Regression

- **Output:** Performance Index prediction

### 3. Titanic Survival Prediction
**File:** `titanic_survival.ipynb`

Predicts passenger survival on the Titanic using historical data and classification algorithms.

**Features:**
- **Input Variables (8 features):**
  - Pclass (Passenger Class)
  - Sex
  - Age
  - SibSp (Siblings/Spouses aboard)
  - Parch (Parents/Children aboard)
  - Fare
  - Embarked (Port of embarkation)
  - CabinDeck

- **Models Used:**
  - Logistic Regression
  - Random Forest Classifier

- **Output:** Binary classification (Survived/Not Survived)

## 🛠️ Installation & Setup

### Prerequisites
```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

### Required Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
```

## 📊 Usage

### Diabetes Prediction
```python
# Example usage
def DiabetesPredictFunction(input_data, model):
    input_array = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    return "Diabetic" if prediction == 1 else "Not Diabetic"

# Input format: [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
sample_input = [2, 120, 70, 30, 100, 25.0, 0.5, 30]
result = DiabetesPredictFunction(sample_input, trained_model)
```

### Titanic Survival Prediction
```python
def SurvivedPeople(input_data, model):
    input_array = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    return "Survived" if prediction == 1 else "Not Survived"

# Interactive input
user_input = input("Enter 8 values separated by space: ")
user_input = tuple(map(float, user_input.split(' ')))
result = SurvivedPeople(user_input, model)
print("Prediction:", result)
```

## 🔧 Data Preprocessing

### Missing Value Handling
- **KNN Imputation:** Uses 5 nearest neighbors to impute missing values
- Maintains data integrity while filling gaps in the dataset

### Feature Engineering
- Proper encoding of categorical variables
- Scaling of numerical features where necessary
- Train-test split for model validation

## 📈 Model Performance

Each project implements multiple algorithms to compare performance:

- **Classification Models:**
  - Logistic Regression (baseline)
  - Random Forest Classifier (ensemble method)

- **Regression Models:**
  - Linear Regression for continuous predictions

## 🎯 Key Features

- **Modular Design:** Each prediction function is self-contained and reusable
- **Interactive Input:** User-friendly input mechanisms for real-time predictions
- **Data Quality:** Robust preprocessing pipeline with missing value handling
- **Multiple Models:** Implementation of various ML algorithms for comparison
- **Clear Output:** Human-readable prediction results

## 📝 Dataset Requirements

### Diabetes Dataset
- **Format:** CSV file with 9 columns (8 features + 1 target)
- **Size:** Recommended 500+ samples for reliable training
- **Path:** Update the file path in the notebook as needed

### Student Performance Dataset
- **Format:** CSV with performance metrics
- **Features:** Academic and lifestyle factors

### Titanic Dataset
- **Format:** Standard Titanic dataset with passenger information
- **Features:** Passenger demographics and ticket details

## 🚀 Getting Started

1. **Clone the repository**
2. **Install dependencies** using the requirements above
3. **Update dataset paths** in the notebooks
4. **Run the Jupyter notebooks** to train models
5. **Use prediction functions** for new data

## 📋 Future Enhancements

- [ ] Add model evaluation metrics visualization
- [ ] Implement cross-validation
- [ ] Add hyperparameter tuning
- [ ] Create a unified prediction interface
- [ ] Add model persistence (save/load functionality)
- [ ] Implement feature importance analysis

## 🤝 Contributing

Feel free to contribute by:
- Adding new prediction models
- Improving existing algorithms
- Enhancing data preprocessing
- Adding visualization features

## 📄 License

This project is open source and available under the MIT License.

---

**Note:** Make sure to update dataset file paths and adjust model parameters based on your specific requirements and data characteristics.
