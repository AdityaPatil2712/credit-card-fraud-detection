# 💳 Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![GitHub Repo stars](https://img.shields.io/github/stars/AdityaPatil2712/credit-card-fraud-detection?style=social)
![GitHub issues](https://img.shields.io/github/issues/AdityaPatil2712/credit-card-fraud-detection)

A **machine learning project** to detect credit card fraud. Includes data preprocessing, feature engineering, model training, evaluation, explainability, and a **Streamlit demo** for real-time predictions.

---

##  Project Overview

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
1) data: # Place dataset CSV here after download

    1.1) creditcard.csv
3) src:
   
   2.1) data_processing.py # Load & preprocess functions
   
   2.2) train_model.py # Training & evaluation pipeline
   
   2.3) predict.py # Predict on new samples
   
   2.4) app_streamlit.py # Streamlit demo
   
5) models:
   
   3.1)model.joblib # Trained model
   
7) notebooks:
   
   4.1) eda_and_training.ipynb # Optional Jupyter notebook for EDA & experiments
   
9) .gitignore
    
11) requirements.txt
    
13) README.md


---

## Dataset

Dataset used: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  

>  **Do not upload the dataset directly** (GitHub file size limit). Please download it from Kaggle and place it in the `data/` folder.

**Steps:**
1. Download the `creditcard.csv` file from the Kaggle link above.  
2. Place it in the `data/` folder.  

---

##  Installation & Usage

1. **Clone the repository**
```bash
git clone https://github.com/AdityaPatil2712/credit-card-fraud-detection.git
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

##  Results:
Accuracy: 99%+ (with imbalanced dataset handling)
F1-Score: 0.95+
ROC-AUC: 0.98
Results may vary depending on model and parameters.

##  Tech Stack
Python 3.x
Pandas, NumPy, Scikit-learn, XGBoost
Matplotlib, Seaborn, SHAP
Streamlit
Joblib (model saving/loading)

##  License
MIT License – see LICENSE file for details.

Contributions & Contact

Open for contributions and feedback!
LinkedIn: Linkedin.com/in/aditya-patil-6596b02b7/
Email: adityapatil27122003@gmail.com
