import streamlit as st
import joblib
import numpy as np
import docx
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure NLTK stopwords data is available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Ensure NLTK wordnet data is available
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# ------------------------- Utility Functions -------------------------

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')
    
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\b\w*\d\w*\b', '', text)
    tokens = tokenizer.tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1 and t.strip().isalpha()]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith('.pdf'):
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ''.join([page.extract_text() or '' for page in reader.pages])
        return text
    elif uploaded_file.name.endswith('.docx'):
        doc = docx.Document(uploaded_file)
        text = '\n'.join([para.text for para in doc.paragraphs])
        return text
    else:
        return ""

def load_model_and_vectorizer(model_name):
    normalized_name = model_name.lower().replace(" ", "_")
    model_path = f"models/{normalized_name}_model.pkl"
    vectorizer_path = "models/tfidf_vectorizer.pkl"
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

# ------------------------- Streamlit UI -------------------------

st.set_page_config(page_title="AI vs Human Text Detector", layout="wide")
st.title("🧠 AI vs Human Text Detector")
st.markdown("""
Upload a **PDF**, **Word Document**, or **paste your own text** to check whether it was written by a human or AI.  
Choose from three trained models: **SVM**, **Decision Tree**, and **AdaBoost**.
""")

# Sidebar - select model
model_choice = st.sidebar.selectbox("Choose a model", ["SVM", "Decision Tree", "AdaBoost"])
model, vectorizer = load_model_and_vectorizer(model_choice)

# Upload or paste text
uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])
text_input = st.text_area("Or paste your text here")

# Predict
if st.button("Detect Author"):
    if uploaded_file or text_input:
        raw_text = extract_text_from_file(uploaded_file) if uploaded_file else text_input
        cleaned_text = preprocess_text(raw_text)
        prediction = model.predict([cleaned_text])[0]
        proba = model.predict_proba([cleaned_text])[0]
        
        label = "🧑 Human" if prediction == 1 else "🤖 AI"
        confidence = np.max(proba) * 100
        
        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: {confidence:.2f}%")
    else:
        st.warning("Please upload a file or paste some text to analyze.")
