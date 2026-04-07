import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from preprocess import clean_text
from model import get_models

# -------------------------
# Load dataset
# -------------------------
# NOTE: Dataset not included in repo (see README)
df = pd.read_csv("data/cyberbullying.csv")

print("First 5 rows:")
print(df.head())

# -------------------------
# Preprocessing
# -------------------------
df['text'] = df['text'].apply(clean_text)

# -------------------------
# Train/Test split
# -------------------------
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# TF-IDF Vectorization
# -------------------------
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -------------------------
# Model Training & Evaluation
# -------------------------
models = get_models()

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, pos_label=1))
    print("Recall:", recall_score(y_test, y_pred, pos_label=1))
    print("F1-score:", f1_score(y_test, y_pred, pos_label=1))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))