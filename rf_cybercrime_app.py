import streamlit as st
import pandas as pd
import numpy as np
import re as regex
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Cybercrime Detection", layout="wide")
st.title("🕵️ Cybercrime Detection using Random Forest")

# Session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None

# Dataset upload and training
st.subheader("📂 Upload Dataset for Training")
uploaded_file = st.file_uploader("Upload `dataset.txt` file (separator = @)", type=["txt"])

if uploaded_file:
    dataset = pd.read_csv(uploaded_file, sep='@')
    st.success("✅ Dataset Loaded")
    st.dataframe(dataset.head())

    # Preprocessing
    st.subheader("🪩 Data Preprocessing and Vectorization")
    le = LabelEncoder()
    dataset['label'] = pd.Series(le.fit_transform(dataset['label'].astype(str)))
    st.session_state.label_encoder = le

    textdata = []
    labels = []
    for i in range(len(dataset)):
        text = dataset.iloc[i]['payload']
        text = regex.sub(r'[^a-zA-Z\s]+', '', text)
        textdata.append(text)
        labels.append(dataset.iloc[i]['label'])

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(textdata).toarray()
    X = normalize(X)
    Y = np.array(labels)
    st.session_state.vectorizer = vectorizer

    # Model training
    st.subheader("🧠 Model Training")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, average='macro') * 100
    rec = recall_score(y_test, y_pred, average='macro') * 100
    f1 = f1_score(y_test, y_pred, average='macro') * 100

    st.success("✅ Model Trained Successfully!")
    st.metric("🎯 Accuracy", f"{acc:.2f}%")
    st.metric("🎯 Precision", f"{prec:.2f}%")
    st.metric("🎯 Recall", f"{rec:.2f}%")
    st.metric("🎯 F1 Score", f"{f1:.2f}%")

    # Save model
    os.makedirs("model", exist_ok=True)
    with open("model/rf.txt", 'wb') as f:
        pickle.dump(model, f)
    st.session_state.model = model
    st.success("✅ Model saved to `model/rf.txt`")

# Prediction on new data
st.subheader("🔍 Predict Cybercrime on New Payloads")
predict_file = st.file_uploader("Upload new data file (1 column: `payload`)", type=["csv"], key="predict_file")

if predict_file and st.session_state.model and st.session_state.vectorizer and st.session_state.label_encoder:
    predict_data = pd.read_csv(predict_file)
    if 'payload' not in predict_data.columns:
        st.error("❌ The CSV must contain a column named 'payload'")
    else:
        clean_data = []
        for i in range(len(predict_data)):
            text = predict_data.iloc[i]['payload']
            text = regex.sub(r'[^a-zA-Z\s]+', '', text)
            clean_data.append(text)

        X_new = st.session_state.vectorizer.transform(clean_data).toarray()
        X_new = normalize(X_new)

        preds = st.session_state.model.predict(X_new)
        decoded_preds = st.session_state.label_encoder.inverse_transform(preds)

        st.subheader("📊 Predictions:")
        for i, prediction in enumerate(decoded_preds):
            st.write(f"Sample {i+1} → `{prediction}`")

# Optional label analytics
st.subheader("📊 Ground Truth Label Analytics (Optional)")
label_file = st.file_uploader("📄 Upload a label file (1 column named 'label')", type=["csv"], key="label_upload")

if label_file:
    label_df = pd.read_csv(label_file)
    if 'label' not in label_df.columns:
        st.error("❌ Your CSV must contain a column named `label`")
    else:
        st.success("✅ Label File Loaded")
        st.write(label_df['label'].value_counts())

        # Count and percentage
        label_counts = label_df['label'].value_counts()
        total = label_counts.sum()

        for label, count in label_counts.items():
            percent = (count / total) * 100
            st.write(f"🧶 `{label}`: {count} records ({percent:.2f}%)")

        # Pie Chart
        fig1, ax1 = plt.subplots()
        ax1.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)

        # Bar Chart
        fig2, ax2 = plt.subplots()
        sns.barplot(x=label_counts.index, y=label_counts.values, palette="coolwarm", ax=ax2)
        ax2.set_title("Label Distribution")
        ax2.set_xlabel("Label")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)
