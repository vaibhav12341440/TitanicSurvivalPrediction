# TitanicSurvivalPrediction
The Titanic Survival Prediction is a classification problem where we develop a Machine Learning (ML) model to predict whether a passenger survived the Titanic disaster based on available passenger data. The dataset includes key features like age, gender, ticket class, fare, cabin information, and more.

The goal is to build a well-trained model with high accuracy using different ML algorithms and evaluate its performance.

Key Features of the Project : 
1️⃣ Data Collection & Understanding
Dataset: Titanic dataset from Kaggle (train.csv & test.csv).

Features include:

PassengerId – Unique ID

Survived – Target variable (0 = No, 1 = Yes)

Pclass – Ticket class (1st, 2nd, 3rd)

Sex – Gender (male/female)

Age – Passenger’s age

SibSp – Number of siblings/spouses aboard

Parch – Number of parents/children aboard

Ticket – Ticket number

Fare – Ticket price

Cabin – Cabin number (contains many missing values)

Embarked – Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

2️⃣ Data Preprocessing & Feature Engineering
Handle Missing Values

Fill missing values using .ffill(), .bfill(), or imputation techniques (mean/median/mode).

Drop irrelevant columns like PassengerId, Ticket, and Cabin (if too many missing values).

Convert Categorical Data

Convert Sex into numeric (0 = Female, 1 = Male).

One-hot encode Embarked (C, Q, S).

Feature Scaling & Normalization

Use StandardScaler() or MinMaxScaler() for numerical features (Age, Fare).

3️⃣ Model Selection & Training
Splitting Data

train_test_split() to divide data into training (80%) & testing (20%) sets.

Machine Learning Models

Logistic Regression (Baseline model)

Decision Tree Classifier

Random Forest Classifier (Ensemble learning)

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Gradient Boosting (XGBoost, LightGBM, CatBoost)

4️⃣ Model Evaluation
Performance Metrics

Accuracy Score (accuracy_score())

Confusion Matrix (confusion_matrix())

Precision, Recall, F1-score (classification_report())

ROC Curve & AUC Score (roc_auc_score())

Hyperparameter Tuning

Use GridSearchCV() or RandomizedSearchCV() to find the best model parameters.

5️⃣ Deployment & Final Steps
Save the trained model using joblib or pickle.

Create a web app using Flask/Streamlit to input passenger details and predict survival.

Push code to GitHub with a structured README.md.

You have uploaded a **Jupyter Notebook (`.ipynb`) file** for **Titanic Survival Prediction**. I will now analyze the code and explain it to you. Let me process the file and provide details.

Your **Titanic Survival Prediction Jupyter Notebook** contains multiple steps for **data preprocessing, model training, and evaluation**. Below is a breakdown of what each section of your code does.

---

## **🔹 Code Explanation**
### **1️⃣ Importing Libraries**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```
- `numpy`: For numerical operations.
- `pandas`: For handling dataframes.
- `matplotlib.pyplot`: For visualizing the dataset.

---

### **2️⃣ Loading the Titanic Dataset**
```python
titanic_data = pd.read_csv('train.csv')
titanic_data
```
- Reads the Titanic dataset (`train.csv`) into a **Pandas DataFrame**.

---

### **3️⃣ Stratified Splitting of Data**
```python
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_indices, test_indices in split.split(titanic_data, titanic_data[["Survived", "Pclass", "Sex"]]):
    strat_train_set = titanic_data.loc[train_indices]
    strat_test_set = titanic_data.loc[test_indices]
```
- **Why?**: Ensures that training and test sets maintain the same proportion of **Survived**, **Pclass**, and **Sex**.
- `StratifiedShuffleSplit()`: Helps prevent **data imbalance** when splitting into training and test sets.

---

### **4️⃣ Visualizing Data Distribution**
```python
plt.subplot(1,2,1)
strat_train_set['Survived'].hist()
strat_train_set['Pclass'].hist()

plt.subplot(1,2,2)
strat_test_set['Survived'].hist()
strat_test_set['Pclass'].hist()
plt.show()
```
- Plots histograms for the **Survived** and **Pclass** features to compare distributions before and after splitting.

---

### **5️⃣ Checking Data Information**
```python
strat_train_set.info()
```
- Displays **column types**, **missing values**, and **dataset statistics**.

---

### **6️⃣ Custom Transformer – Age Imputation**
```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

class AgeImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        imputer = SimpleImputer(strategy="mean")
        X['Age'] = imputer.fit_transform(X[['Age']])
        return X
```

