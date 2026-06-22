"""
Streamlit app for the NLP Sentiment Analysis project.
Run after training:
    streamlit run app.py
"""

from pathlib import Path
import sys
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from preprocessing import preprocess_text

MODEL_PATH = ROOT / "models" / "best_sentiment_model.pkl"

st.set_page_config(page_title="NLP Sentiment Analysis", page_icon="💬", layout="centered")
st.title("💬 NLP Sentiment Analysis")
st.write("Enter a product review. The model will classify it as Positive or Negative.")

if not MODEL_PATH.exists():
    st.error("Model not found. Please run: python src/train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)
review = st.text_area("Product Review", "This product is not good and not worth the price.", height=140)

if st.button("Predict Sentiment"):
    clean = preprocess_text(review)
    prediction = model.predict([clean])[0]
    st.subheader("Prediction")
    if prediction == "positive":
        st.success("Positive Review ✅")
    else:
        st.error("Negative Review ❌")
    st.caption(f"Preprocessed text: {clean}")
