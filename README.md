# Ibm-project

# 💼 Salary Prediction Web Application

This project is a **machine learning pipeline** built to predict whether a person earns above or below $50K annually based on various personal and professional attributes. It uses **Logistic Regression** as the core model and is deployed as an interactive **Streamlit** web application.

---

## 📊 Objective

The main goal of this project is to:
- Apply data preprocessing techniques on structured data
- Train a classification model to predict income levels
- Serve the model via a user-friendly web interface using Streamlit
- Demonstrate the end-to-end deployment of a machine learning project

---

## 🧠 Dataset

We used the **Adult Income Dataset (also known as Census Income Dataset)** from the UCI Machine Learning Repository:
- Link: [UCI Repository](https://archive.ics.uci.edu/ml/datasets/adult)
- Number of Instances: ~48,842
- Features include: age, workclass, education, marital-status, occupation, race, sex, hours-per-week, etc.
- Target: `<=50K` or `>50K`

---

## 🛠️ Features Implemented

- Full preprocessing pipeline using `ColumnTransformer`:
  - `StandardScaler` for numerical columns
  - `OneHotEncoder` for categorical columns
- Trained using `LogisticRegression` classifier
- Split dataset using stratified train/test split
- Saved model (`.pkl`) and preprocessor pipeline
- Developed Streamlit UI with dropdowns and sliders for inputs
- Displays prediction: **Will the person earn >50K or <=50K?**

---

## 📂 Project Structure

```
.
├── app.py                        # Streamlit app file
├── code.ipynb                    # Jupyter Notebook (EDA + Model Training)
├── salary_model.pkl              # Trained Logistic Regression model
├── salary_preprocessor.pkl       # Saved preprocessor pipeline
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## 🧪 Training Summary

- **Model Used**: Logistic Regression
- **Preprocessing**:
  - Numerical: StandardScaler
  - Categorical: OneHotEncoder
- **Evaluation**:
  - Accuracy: ~85%
  - F1-score: Balanced across classes
- **Libraries**: scikit-learn, pandas, numpy, streamlit, joblib

---

## 🚀 How to Run the Project

### 1️⃣ Install Required Packages

```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

Make sure the following files are in the **same directory**:
- `app.py`
- `salary_model.pkl`
- `salary_preprocessor.pkl`

---

## 📝 Example Input Fields in UI

- Age: slider
- Education: dropdown
- Occupation: dropdown
- Hours-per-week: slider
- Gender: dropdown
- Native Country: dropdown
- Workclass, Marital status, etc.

---

## 📷 Screenshots

*Add screenshots here after launching the app locally.*

---

## 📦 Requirements

Create a `requirements.txt` file with the following content:

```
streamlit
scikit-learn
pandas
numpy
joblib
```

---

## 🎓 Learning Outcomes

- Understanding of classification pipelines
- Handling both categorical and numerical data
- Saving and loading ML models and preprocessors
- Deploying ML apps with Streamlit

---

## 🙋‍♂️ About the Author

**Akash**  
Aspiring Data Engineer | Passionate about real-world ML apps  
GitHub: [your-username]  
LinkedIn: [your-link]

---

## 📜 License

This project is licensed for educational and demonstration purposes.

