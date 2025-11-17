import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
import joblib

# Load + basic clean
data = pd.read_excel("D:\\Semesters\\sem2\\ISTE_612_InfoReterival\\project\\wikipedia_bias_detection\\data\\wikipedia_10000_bias_dataset.xlsx")
data = data.drop(columns=["Unnamed: 0", "title", "political_bias"], errors="ignore")
data["text"] = data["text"].astype(str)

X = data["text"]
y = data["bias_label"]

# Split
X_train_text, X_test_text, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# TF-IDF
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words="english")
tfidf.fit(X_train_text)

X_train_tfidf = tfidf.transform(X_train_text)
X_test_tfidf  = tfidf.transform(X_test_text)

# Logistic Regression + calibration
base_lr = LogisticRegression(class_weight="balanced", max_iter=2000, n_jobs=-1)

calib_lr = CalibratedClassifierCV(estimator=base_lr, method="isotonic", cv=5)

calib_lr.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = calib_lr.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# Save everything needed for inference
joblib.dump(tfidf, "models/tfidf_vectorizer.joblib")
joblib.dump(calib_lr, "models/bias_lr_calibrated.joblib")
print("Saved tfidf_vectorizer.joblib and bias_lr_calibrated.joblib")
