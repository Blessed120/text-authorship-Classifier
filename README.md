# text-authorship-Classifier
This is an ai-human-text detector

# 🧠 AI vs Human Text Detection

This is a machine learning web application that detects whether a given text was written by a **human** or **AI**. It supports file uploads (PDF/DOCX) or direct text input and uses three classification models: **SVM**, **Decision Tree**, and **AdaBoost**.

---

## 🚀 Features
- You can Upload a file and the system will analyze `.pdf`, `.docx`, or typed/pasted text
- Select from three pre-trained models
- Get prediction and confidence score



## 🛠 Setup Instructions

- Clone the repo or download the project folder.
- (Optional) Create a virtual environment:

```bash

- python -m venv venv
- source venv/bin/activate  # or venv\\Scripts\\activate

# on Windows

pip install -r requirements.txt

streamlit run app.py
