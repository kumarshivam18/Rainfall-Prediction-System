# Weather Prediction: Rain on Tomorrow

This project demonstrates a machine learning model to predict whether it will rain tomorrow based on various weather attributes. The model uses the Random Forest Classifier and is built with the scikit-learn library.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Examples](#examples)
- [Acknowledgements](#acknowledgements)

## Installation

### Prerequisites

Ensure you have Python 3.x installed. You can download it from [python.org](https://www.python.org/).

### Dependencies

Install the required packages using pip:

```sh
pip install numpy pandas scikit-learn matplotlib
```

## Usage

1. **Place the dataset file `weatherAUS.csv` in the project directory.**

2. **Run the script:**

   ```sh
   python main.py
   ```

## Project Structure

- `main.py`: Main script for loading data, preprocessing, training, and evaluating the model.
- `README.md`: Project documentation.
- `weatherAUS.csv`: Input dataset file (make sure to download it and place it in the project directory).

## Dataset

The dataset used in this project is the "Rain in Australia" dataset from Kaggle. It contains daily weather observations from numerous Australian weather stations. The target variable is whether or not it rained the next day.

### Input File

- **weatherAUS.csv**: This file should be placed in the project directory. It contains the weather data used for training and testing the model. You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package).

## Methodology

### 1. Import Libraries

Essential libraries are imported for data manipulation, machine learning, and visualization.

```python
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
```

### 2. Load and Preprocess Data

The dataset is loaded, and missing values are handled using the `SimpleImputer` from scikit-learn.

```python
dataset = pd.read_csv('weatherAUS.csv')
X = dataset.iloc[:, [1,2,3,4,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]].values
Y = dataset.iloc[:, -1].values
Y = Y.reshape(-1, 1)

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
X = imputer.fit_transform(X)
Y = imputer.fit_transform(Y)
```

### 3. Encode Categorical Data

Label encoding is applied to convert categorical variables into numerical values.

```python
le1 = LabelEncoder()
X[:, 0] = le1.fit_transform(X[:, 0])
le2 = LabelEncoder()
X[:, 4] = le2.fit_transform(X[:, 4])
le3 = LabelEncoder()
X[:, 6] = le3.fit_transform(X[:, 6])
le4 = LabelEncoder()
X[:, 7] = le4.fit_transform(X[:, 7])
le5 = LabelEncoder()
X[:, -1] = le5.fit_transform(X[:, -1])
le6 = LabelEncoder()
Y = le6.fit_transform(Y)
```

### 4. Feature Scaling

Standard scaling is applied to standardize the features.

```python
sc = StandardScaler()
X = sc.fit_transform(X)
```

### 5. Split Data

The dataset is split into training and testing sets.

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
```

### 6. Train the Model

The Random Forest Classifier is trained on the training set.

```python
classifier = RandomForestClassifier(n_estimators=100, random_state=0)
classifier.fit(X_train, Y_train)
```

### 7. Evaluate the Model

The model is evaluated on the test set, and predictions are compared with actual values.

```python
y_pred = classifier.predict(X_test)
y_pred = le6.inverse_transform(y_pred)
Y_test = le6.inverse_transform(Y_test)

accuracy = accuracy_score(Y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
```

### 8. Save Predictions

The predictions and actual values are saved to a CSV file.

```python
df = np.concatenate((Y_test.reshape(-1, 1), y_pred.reshape(-1, 1)), axis=1)
dataframe = pd.DataFrame(df, columns=['Rain on Tomorrow', 'Prediction of Rain'])
dataframe.to_csv('prediction.csv', index=False)
```

## Examples

**Example Data Output:**

```
[['Yes' 'No']
 ['Yes' 'No']
 ['No' 'No']
 ...
 ['Yes' 'No']
 ['No' 'No']
 ['No' 'No']]
```

## Acknowledgements

This project uses the scikit-learn library for machine learning. Special thanks to the Australian Government Bureau of Meteorology for providing the weather dataset.
