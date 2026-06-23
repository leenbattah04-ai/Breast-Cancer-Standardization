import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
dataset = sklearn.datasets.load_breast_cancer()

# Convert to DataFrame
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

print(df.head())
print(df.shape)

# Features and target
X = df
Y = dataset.target

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=3
)

print(X.shape, X_train.shape, X_test.shape)

# Standardization
scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Check results
print(X_train_scaled.std())
print(X_test_scaled.std())