pipeline {
    agent any

    environment {
        VENV_DIR = "venv"
    }

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/Blessed120/text-authorship-Classifier.git'
            }
        }

        stage('Set up Python Environment') {
            steps {
                sh '''
                    python3 -m venv $VENV_DIR
                    source $VENV_DIR/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Run Streamlit App') {
            steps {
                sh '''
                    source $VENV_DIR/bin/activate
                    nohup streamlit run app.py --server.port=8501 &
                '''
            }
        }
    }

    post {
        always {
            echo "✅ Pipeline execution complete."
        }
    }
}
