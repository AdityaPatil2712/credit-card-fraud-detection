# 💳 Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![GitHub Repo stars](https://img.shields.io/github/stars/<your-username>/credit-card-fraud-detection?style=social)
![GitHub issues](https://img.shields.io/github/issues/<your-username>/credit-card-fraud-detection)

A **machine learning project** to detect credit card fraud. Includes data preprocessing, feature engineering, model training, evaluation, explainability, and a **Streamlit demo** for real-time predictions.

---

## 🚀 Project Overview

Credit card fraud is a growing problem worldwide. This project demonstrates a **complete ML workflow** for detecting fraudulent transactions:

- **Data preprocessing:** Scaling, handling imbalanced datasets  
- **Feature engineering & selection**  
- **Model training:** Logistic Regression, Random Forest, XGBoost  
- **Evaluation:** Confusion matrix, precision, recall, F1, ROC-AUC  
- **Model explainability:** SHAP visualizations  
- **Demo:** Interactive Streamlit app for predictions  

---

## 📂 Project Structure

credit-card-fraud-detection/
├─ data/ # Place dataset CSV here after download
│ └─ creditcard.csv
├─ src/
│ ├─ data_processing.py # Load & preprocess functions
│ ├─ train_model.py # Training & evaluation pipeline
│ ├─ predict.py # Predict on new samples
│ └─ app_streamlit.py # Streamlit demo
├─ models/
│ └─ model.joblib # Trained model
├─ notebooks/
│ └─ eda_and_training.ipynb # Optional Jupyter notebook for EDA & experiments
├─ .gitignore
├─ requirements.txt
└─ README.md


---

## 📖 Dataset

Dataset used: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  

> ⚠️ **Do not upload the dataset directly** (GitHub file size limit). Please download it from Kaggle and place it in the `data/` folder.

**Steps:**
1. Download the `creditcard.csv` file from the Kaggle link above.  
2. Place it in the `data/` folder.  

---

## 🛠️ Installation & Usage

1. **Clone the repository**
```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection

Create virtual environment & install dependencies:
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt

Run Jupyter notebook (optional):
jupyter notebook notebooks/eda_and_training.ipynb

Run Streamlit demo:
streamlit run src/app_streamlit.py

📈 Results

Accuracy: 99%+ (with imbalanced dataset handling)

F1-Score: 0.95+

ROC-AUC: 0.98

Results may vary depending on model and parameters.

🔧 Tech Stack

Python 3.x

Pandas, NumPy, Scikit-learn, XGBoost

Matplotlib, Seaborn, SHAP

Streamlit

Joblib (model saving/loading)

📝 License

MIT License – see LICENSE file for details.

🌟 Contributions & Contact

Open for contributions and feedback!

LinkedIn: Linkedin.com/in/aditya-patil-6596b02b7/

Email: adityapatil27122003@gmail.com