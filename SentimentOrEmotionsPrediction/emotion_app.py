import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Load Model & Vectorizer
# --------------------------
model = pickle.load(open("emotion_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Emotion labels
label_map = {
    0: "Sadness",
    1: "Anger",
    2: "Love",
    3: "Surprise",
    4: "Fear",
    5: "Joy"
}

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Emotion Detector", page_icon="💬", layout="centered")

st.title("💬 Emotion Detection App")
st.write("Enter a sentence and let the ML model detect the emotion.")

# User Input
text = st.text_area("Enter text:", placeholder="Type something like: I am very happy today!")

# Prediction Button
if st.button("Predict Emotion"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform text
        X = vectorizer.transform([text])

        # Predict class and probability
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        # Output
        st.subheader("Predicted Emotion:")
        st.success(label_map[prediction])

        # Show probabilities
        st.subheader("Emotion Probability Distribution")

        fig, ax = plt.subplots()
        ax.bar(label_map.values(), probabilities)
        ax.set_xlabel("Emotion")
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Confidence")
        plt.xticks(rotation=45)

        st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Machine Learning By Abhishek Gorya")
#Run with "streamlit run emotion_app.py"