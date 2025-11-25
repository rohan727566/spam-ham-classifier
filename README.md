\# 🚀 Spam/Ham Email Classification System



\[!\[Python 3.11.5](https://img.shields.io/badge/python-3.11.5-blue.svg)](https://www.python.org/downloads/)

\[!\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

\[!\[CI](https://github.com/rohan727566/spam-ham-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/rohan727566/spam-ham-classifier/actions)



Production-grade NLP system for spam/ham email classification using Naive Bayes and Natural Language Processing.



\## 📋 Features



\- ✅ Text preprocessing with NLTK (stopwords, lemmatization)

\- ✅ TF-IDF vectorization

\- ✅ Multinomial Naive Bayes classifier

\- ✅ FastAPI REST API

\- ✅ Interactive Web UI

\- ✅ Docker containerization

\- ✅ CI/CD with GitHub Actions

\- ✅ Deployment ready (Render)



\## 🏗️ Project Structure



spam-ham-classifier/

├── src/spam\_classifier/ # Core ML modules

├── tests/ # Unit tests

├── web/ # Frontend UI

├── scripts/ # Utility scripts

├── data/ # Dataset storage

├── model/ # Trained models

├── docs/ # Documentation \& plots

└── .github/workflows/ # CI/CD pipelines



\## 🚀 Quick Start



\### Local Development



Clone repository

git clone https://github.com/rohan727566/spam-ham-classifier.git

cd spam-ham-classifier



Create virtual environment

python -m venv venv

venv\\Scripts\\activate # Windows



source venv/bin/activate # Linux/Mac

Install dependencies

pip install -r requirements.txt



Train model

python -m src.spam\_classifier.train --dataset data/SMSSpamCollection.tsv



Run API server

python -m src.spam\_classifier.server



\### Docker



Build image

docker build -t spam-classifier .



Run container

docker run -p 8000:8000 spam-classifier



\## 📊 Model Performance



| Metric    | Score |

|-----------|-------|

| Accuracy  | TBD   |

| Precision | TBD   |

| Recall    | TBD   |

| F1-Score  | TBD   |



\## 🛠️ Tech Stack



\- \*\*ML/NLP\*\*: Python 3.11.5, scikit-learn, NLTK, pandas

\- \*\*API\*\*: FastAPI, uvicorn

\- \*\*Frontend\*\*: HTML5, JavaScript, CSS3

\- \*\*Deployment\*\*: Docker, Render

\- \*\*CI/CD\*\*: GitHub Actions



\## 📝 Author



\*\*Rohan Kumar\*\*  

AI Sep 2024 Batch - Training Project (27th Oct)



\## 📄 License



MIT License - see \[LICENSE](LICENSE) file for details.



