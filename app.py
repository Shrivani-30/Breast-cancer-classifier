import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("data/breast_cancer.csv")

# Features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Streamlit interface
st.title("Breast Cancer Classifier")

st.write("Breast Cancer Classification using Logistic Regression")

st.write("Enter the feature values below:")

feature_values = []

for column in X.columns:
    value = st.number_input(column, value=0.0)
    feature_values.append(value)

if st.button("Predict"):
    input_data = pd.DataFrame([feature_values], columns=X.columns)
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    if prediction == 0:
        st.success("Prediction: Malignant")
    else:
        st.success("Prediction: Benign")
