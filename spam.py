import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# STEP 1: Load and Preprocess Dataset
# -------------------------------
print("📥 Loading Dataset...")

try:
    df = pd.read_csv("spam_ham_dataset.csv", encoding='ISO-8859-1')
except FileNotFoundError:
    print("❌ File 'spam_ham_dataset.csv' not found. Please check the file path.")
    exit()

df = df.rename(columns={'label': 'label', 'text': 'email_text'})
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

if df.isnull().sum().any():
    print("⚠ Missing values found. Dropping...")
    df = df.dropna()

print("✅ Dataset Loaded and Preprocessed!")
print("Unique Labels:", df['label'].unique())

# -------------------------------
# STEP 2: Split Data
# -------------------------------
print("\n✂ Splitting Data...")

X = df['email_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")

# -------------------------------
# STEP 3: Vectorization
# -------------------------------
print("\n🧠 Applying TF-IDF Vectorization...")

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("✅ Text Vectorization Complete!")

# -------------------------------
# STEP 4: Train SVM Classifier
# -------------------------------
print("\n⚙️ Training SVM Classifier...")

svm_clf = SVC(kernel='linear')
svm_clf.fit(X_train_vec, y_train)

print("✅ Training Complete!")

# -------------------------------
# STEP 5: Evaluate Model
# -------------------------------
print("\n📊 Evaluating Model...")

y_pred = svm_clf.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\n✅ Accuracy: {accuracy*100:.2f}%")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", report)

# -------------------------------
# STEP 6: Spam Alert Function
# -------------------------------
def spam_alert(email_text):
    email_vec = vectorizer.transform([email_text])
    prediction = svm_clf.predict(email_vec)[0]
    return "🚨 Spam Alert!" if prediction == 1 else "✅ Legitimate Email"

# -------------------------------
# STEP 7: Apply Function to Entire Dataset
# -------------------------------
print("📤 Applying spam_alert() function to dataset...")

df['spam_check'] = df['email_text'].apply(spam_alert)

df.to_csv("spam_ham_results.csv", index=False)
print("✅ Results saved to spam_ham_results.csv!")

# -------------------------------
# STEP 8: Sample Predictions Like Screenshot
# -------------------------------
print("\n📬 Sample Classified Emails:\n")

# Legitimate examples
print("---- Legitimate Emails (Ham) ----")
ham_samples = df[df['label'] == 0].sample(5, random_state=1)
for i, row in ham_samples.iterrows():
    print(f"📩 {row['email_text'][:100].replace(chr(10), ' ')}...")
    print(f"Prediction: {row['spam_check']}\n")

# Spam examples
print("---- Spam Emails ----")
spam_samples = df[df['label'] == 1].sample(5, random_state=2)
for i, row in spam_samples.iterrows():
    print(f"📩 {row['email_text'][:100].replace(chr(10), ' ')}...")
    print(f"Prediction: {row['spam_check']}\n")

# -------------------------------
# STEP 9: Manual User Input Section
# -------------------------------
while True:
    print("\n🔎 Test Your Own Email Message!")
    user_input = input("Type your email message (or type 'exit' to quit):\n>>> ")

    if user_input.lower() == 'exit':
        print("👋 Exiting program. Goodbye!")
        break

    result = spam_alert(user_input)
    print("📬 Prediction:", result)
