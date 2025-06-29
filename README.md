# 📧 Spam Email Classifier using Machine Learning

This project is a simple yet powerful machine learning model that can classify emails as **Spam** or **Ham (Legitimate)** using **TF-IDF Vectorization** and a **Support Vector Machine (SVM)** classifier.

---

## 🚀 Features

- Classifies emails as 🚨 **Spam** or ✅ **Ham**
- Preprocesses and vectorizes email text using **TF-IDF**
- Trains a robust **SVM model**
- Provides:
  - Accuracy score
  - Confusion matrix
  - Classification report
- Allows users to input custom emails and test them
- Displays sample classified spam and ham emails
- Saves results to CSV file

---

## 🧠 Technologies Used

- **Python 3.x**
- **Pandas** – for data manipulation
- **Scikit-learn** – for ML modeling, vectorization, and evaluation
- **SVM (Support Vector Machine)** – for classification
- **TF-IDF Vectorizer** – to convert text to numerical features

---

## 📁 Dataset

The dataset used is a CSV file with labeled email texts:
- `label` column: `spam` or `ham`
- `text` column: the actual email content

---

## 🛠 How It Works

1. **Data Loading**  
   Loads and preprocesses the spam-ham dataset.

2. **Data Splitting**  
   Divides the dataset into 80% training and 20% testing sets.

3. **Vectorization**  
   Converts email text into numeric features using `TfidfVectorizer`.

4. **Model Training**  
   Trains a `Support Vector Machine (SVM)` classifier on the training set.

5. **Model Evaluation**  
   Prints the model’s accuracy, confusion matrix, and a detailed classification report.

6. **Real-Time Testing**  
   Accepts user input for real-time spam detection.

7. **Result Saving**  
   Writes spam predictions to `spam_ham_results.csv`.

---

## 📦 How to Run

### 1. Clone the Repo
```bash

