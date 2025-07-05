import streamlit as st
import pandas as pd
import numpy as np
import re as regex
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Cybercrime Detection", layout="wide")
st.title("🕵️‍♀️ Cybercrime Detection using Random Forest")

# ----------------------------------
# Session state setup
# ----------------------------------
if "model" not in st.session_state:
    st.session_state.model = None
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "label_encoder" not in st.session_state:
    st.session_state.label_encoder = None

# ----------------------------------
# 1️⃣  Upload & Train Section
# ----------------------------------
st.subheader("📂 Upload Dataset for Training")
uploaded_file = st.file_uploader(
    "Upload `dataset.txt` file (separator = @)", type=["txt"]
)

if uploaded_file is not None:
    # Load dataset
    dataset = pd.read_csv(uploaded_file, sep="@")
    st.success("✅ Dataset Loaded")
    st.dataframe(dataset.head())

    # Pre‑process
    st.subheader("🧹 Data Preprocessing and Vectorization")
    le = LabelEncoder()
    dataset["label"] = pd.Series(
        le.fit_transform(dataset["label"].astype(str))
    )
    st.session_state.label_encoder = le

    clean_texts, labels = [], []
    for text, lbl in zip(dataset["payload"], dataset["label"]):
        text = regex.sub(r"[^a-zA-Z\s]+", "", text)
        clean_texts.append(text)
        labels.append(lbl)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(clean_texts).toarray()
    X = normalize(X)
    Y = np.array(labels)
    st.session_state.vectorizer = vectorizer

    # Train / Test split & training
    st.subheader("🧠 Model Training")
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, random_state=0)
    model.fit(X_train, y_train)

    # Evaluation metrics
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, average="macro") * 100
    rec = recall_score(y_test, y_pred, average="macro") * 100
    f1 = f1_score(y_test, y_pred, average="macro") * 100

    st.success("✅ Model Trained Successfully!")
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{acc:.2f}%")
    cols[1].metric("Precision", f"{prec:.2f}%")
    cols[2].metric("Recall", f"{rec:.2f}%")
    cols[3].metric("F1 Score", f"{f1:.2f}%")

    # 🔥 Confusion Matrix Heatmap
    st.subheader("📌 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        ax=ax_cm,
    )
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title("Confusion Matrix – Random Forest")
    st.pyplot(fig_cm)

    # Save model for later predictions
    os.makedirs("model", exist_ok=True)
    with open("model/rf.txt", "wb") as f:
        pickle.dump(model, f)
    st.session_state.model = model
    st.success("✅ Model saved to `model/rf.txt`")

# ----------------------------------
# 2️⃣  Prediction Section
# ----------------------------------
st.subheader("🔍 Predict Cybercrime on New Payloads")
predict_file = st.file_uploader(
    "Upload new data file (1 column named `payload`)",
    type=["csv"],
    key="predict_file",
)

if (
    predict_file is not None
    and st.session_state.model is not None
    and st.session_state.vectorizer is not None
    and st.session_state.label_encoder is not None
):
    pred_df = pd.read_csv(predict_file)
    if "payload" not in pred_df.columns:
        st.error("❌ The CSV must contain a column named 'payload'.")
    else:
        texts = pred_df["payload"].apply(
            lambda x: regex.sub(r"[^a-zA-Z\s]+", "", str(x))
        ).tolist()
        X_new = st.session_state.vectorizer.transform(texts).toarray()
        X_new = normalize(X_new)
        preds = st.session_state.model.predict(X_new)
        decoded_preds = st.session_state.label_encoder.inverse_transform(preds)

        st.subheader("📊 Predictions:")
        for idx, pred in enumerate(decoded_preds, 1):
            st.write(f"Sample {idx} → `{pred}`")

# ----------------------------------
# 3️⃣  Optional Ground Truth Analytics
# ----------------------------------
st.subheader("📊 Ground Truth Label Analytics (Optional)")
label_file = st.file_uploader(
    "📄 Upload ground truth label file (1 column named 'label')",
    type=["csv"],
    key="label_upload",
)

if label_file is not None:
    label_df = pd.read_csv(label_file)
    if "label" not in label_df.columns:
        st.error("❌ Your CSV must contain a column named `label`.")
    else:
        st.success("✅ Label file loaded.")
        counts = label_df["label"].value_counts()
        st.write(counts)

        total = counts.sum()
        for lbl, cnt in counts.items():
            st.write(f"🔸 `{lbl}`: {cnt} records ({(cnt/total)*100:.2f}%)")

        # Pie chart
        fig_pie, ax_pie = plt.subplots()
        ax_pie.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
        ax_pie.axis("equal")
        st.pyplot(fig_pie)

        # Bar chart
        fig_bar, ax_bar = plt.subplots()
        sns.barplot(x=counts.index, y=counts.values, palette="coolwarm", ax=ax_bar)
        ax_bar.set_title("Label Distribution")
        ax_bar.set_xlabel("Label")
        ax_bar.set_ylabel("Count")
        st.pyplot(fig_bar)
