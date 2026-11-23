
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")

st.title("Diabetes Prediction Using Logistic Regression")
st.write("A machine learning application to predict diabetes.")

@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error("diabetes.csv file not found. Please add it to the project folder.")
    st.stop()

st.subheader("Dataset Overview")
st.write("Dataset Shape:", df.shape)
st.dataframe(df.head())

target = "Outcome"

if target not in df.columns:
    st.error("The dataset must contain an Outcome column.")
    st.stop()

X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

st.subheader("Model Performance")

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1 Score": f1_score(y_test, y_pred, zero_division=0),
    "ROC-AUC": roc_auc_score(y_test, y_prob)
}

for name, value in metrics.items():
    st.metric(name, f"{value:.3f}")

st.subheader("ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label="Logistic Regression")
ax.plot([0, 1], [0, 1], linestyle="--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
st.pyplot(fig)

st.subheader("Confusion Matrix")
st.write(confusion_matrix(y_test, y_pred))

st.subheader("Predict Diabetes Risk")

user_input = {}

for col in X.columns:
    user_input[col] = st.number_input(
        f"Enter {col}",
        min_value=0.0,
        value=float(X[col].median())
    )

if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.write(f"Predicted diabetes probability: {probability:.2%}")

    if prediction == 1:
        st.warning("Model prediction: Diabetes")
    else:
        st.success("Model prediction: No Diabetes")

    st.caption("This is an educational machine-learning prediction, not a medical diagnosis.")
