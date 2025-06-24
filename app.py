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

# Ensure the presence of NLTK stopwords 
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Ensure the presence of NLTK wordnet 
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Utility Functions

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

# Streamlit UI 

st.set_page_config(page_title="AI vs Human Text Detector", layout="wide")
st.title("🧠 AI vs Human Text Detector")

# Welcome Message for the users
st.markdown("""
<div style='
    background-color: #6f42c1;
    color: white;
    padding: 1.2rem;
    border-radius: 0.5rem;
    border-left: 6px solid #3b237a;
    margin-bottom: 1.5rem;
    font-size: 1.05rem;
'>
    👋 <strong>Welcome!</strong> Paste your content or upload a document to find out if it was written by a human or an AI.  
    Choose your preferred model from the sidebar, then hit <em>Detect Author</em> to get your result!
</div>
""", unsafe_allow_html=True)

st.markdown("""
Upload a **PDF**, **Word Document**, or **paste your own text** to check whether it was written by a human or AI.  
Choose from three trained models: **SVM**, **Decision Tree**, and **AdaBoost**.
""")

# select model needed model
with st.sidebar:
    st.header("⚙️ Model Selection")
    model_choice = st.selectbox("Choose a model", ["SVM", "Decision Tree", "AdaBoost"])
    st.markdown("ℹ️ You can switch between models to compare results.")

model, vectorizer = load_model_and_vectorizer(model_choice)

# Upload and text input section
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("📄 Upload PDF or Word Document", type=["pdf", "docx"])

with col2:
    with st.expander("📋 Or paste text manually"):
        text_input = st.text_area("Enter your text here", height=250)

# Control of the predict section
if st.button("🚀 Detect Author"):
    if uploaded_file or text_input:
        raw_text = extract_text_from_file(uploaded_file) if uploaded_file else text_input
        cleaned_text = preprocess_text(raw_text)
        prediction = model.predict([cleaned_text])[0]
        proba = model.predict_proba([cleaned_text])[0]
        
        label = "🧑 Human" if prediction == 1 else "🤖 AI"
        confidence = np.max(proba) * 100

        # Styling of the result box
        st.markdown(f"""
            <div style='
                padding: 1.2rem;
                background-color: #6f42c1;
                color: white;
                border-radius: 0.5rem;
                border-left: 5px solid #3b237a;
                font-size: 1.1rem;'>
                <strong>Prediction:</strong> {label}  
                <br><strong>Confidence:</strong> {confidence:.2f}%
            </div>
            """, unsafe_allow_html=True)

        st.progress(int(confidence))
    else:
        st.warning("Please upload a file or paste some text to analyze.")

# Here  is the Footer

st.markdown("""<hr style="margin-top: 3rem; margin-bottom: 1rem;">""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; font-size: 0.9rem; color: gray;'>
    Made with ❤️ using <strong>Streamlit</strong>  
    <br>© 2025 AI vs Human Text Detector
</div>
""", unsafe_allow_html=True)
