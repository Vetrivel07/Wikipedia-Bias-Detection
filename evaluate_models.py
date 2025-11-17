# evaluate_models.py
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

from transformers import pipeline
import torch

# ----------------- load data -----------------

DATA_PATH = r"<--- your file path ---->"
data = pd.read_excel(DATA_PATH)

# keep only what we need
data = data.drop(columns=["Unnamed: 0", "title", "political_bias"], errors="ignore")
data["text"] = data["text"].astype(str)

X = data["text"]
y = data["bias_label"]   # 'NEGATIVE' / 'NEUTRAL' / 'POSITIVE'

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ----------------- load trained LR model -----------------

MODEL_DIR = "models"
tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
lr_model = joblib.load(os.path.join(MODEL_DIR, "bias_lr_calibrated.joblib"))

X_test_vec = tfidf.transform(X_test_text)

os.makedirs("static/metrics", exist_ok=True)

classes = list(lr_model.classes_)

# ----------------- 1) LR metrics + confusion matrix -----------------

print("=== Logistic Regression bias model ===")
y_pred_lr = lr_model.predict(X_test_vec)
report_lr = classification_report(y_test, y_pred_lr, labels=classes, digits=4)
print(report_lr)

with open("static/metrics/lr_report.txt", "w", encoding="utf-8") as f:
    f.write(report_lr)

# normalized by TRUE label (row), expressed as percentages
cm_lr = confusion_matrix(y_test, y_pred_lr, labels=classes, normalize="true") * 100.0

plt.figure(figsize=(4, 3))
plt.imshow(cm_lr, interpolation="nearest", cmap="Blues")
# plt.title("Confusion matrix – Logistic Regression")
cbar = plt.colorbar()
cbar.set_label("Percentage (%)", rotation=90)

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel("Predicted")
plt.ylabel("True")

thresh = cm_lr.max() / 2.0
for i in range(cm_lr.shape[0]):
    for j in range(cm_lr.shape[1]):
        plt.text(
            j,
            i,
            f"{cm_lr[i, j]:.2f}",  # show 1 decimal place
            ha="center",
            va="center",
            color="white" if cm_lr[i, j] > thresh else "black",
        )

plt.tight_layout()
plt.savefig("static/metrics/bias_confusion_lr.png", bbox_inches="tight")
plt.close()


# ----------------- 2) Baseline model = BERTweet sentiment model -----------------

device = 0 if torch.cuda.is_available() else -1
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="finiteautomata/bertweet-base-sentiment-analysis",
    truncation=True,
    device=device,
)

def teacher_label(text: str) -> str:
    """
    Map BERTweet labels NEG / NEU / POS
    to our dataset labels NEGATIVE / NEUTRAL / POSITIVE.
    """
    out = sentiment_pipeline(text)[0]["label"]  # 'NEG', 'NEU', 'POS'
    if out == "NEG":
        return "NEGATIVE"
    elif out == "POS":
        return "POSITIVE"
    else:
        return "NEUTRAL"

print("\n=== Baseline model: BERTweet sentiment (teacher) ===")

y_pred_teacher = [teacher_label(t) for t in X_test_text]
report_teacher = classification_report(y_test, y_pred_teacher, labels=classes, digits=4)
print(report_teacher)

with open("static/metrics/teacher_report.txt", "w", encoding="utf-8") as f:
    f.write(report_teacher)

cm_teacher = confusion_matrix(
    y_test, y_pred_teacher, labels=classes, normalize="true"
) * 100.0

plt.figure(figsize=(4, 3))
plt.imshow(cm_teacher, interpolation="nearest", cmap="Greens")
# plt.title("Confusion matrix – BERTweet baseline")
cbar = plt.colorbar()
cbar.set_label("Percentage (%)", rotation=90)

plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel("Predicted")
plt.ylabel("True")

thresh = cm_teacher.max() / 2.0
for i in range(cm_teacher.shape[0]):
    for j in range(cm_teacher.shape[1]):
        plt.text(
            j,
            i,
            f"{cm_teacher[i, j]:.2f}",
            ha="center",
            va="center",
            color="white" if cm_teacher[i, j] > thresh else "black",
        )

plt.tight_layout()
plt.savefig("static/metrics/bias_confusion_teacher.png", bbox_inches="tight")
plt.close()

# ----------------- 3) Linear SVM (LinearSVC) model -----------------

from sklearn.svm import LinearSVC

print("\n=== Linear SVM (SVC) model ===")

svm_model = LinearSVC()
svm_model.fit(tfidf.transform(X_train_text), y_train)

y_pred_svm = svm_model.predict(X_test_vec)

report_svm = classification_report(y_test, y_pred_svm, labels=classes, digits=4)
print(report_svm)

with open("static/metrics/svm_report.txt", "w", encoding="utf-8") as f:
    f.write(report_svm)

cm_svm = confusion_matrix(y_test, y_pred_svm, labels=classes, normalize="true") * 100

plt.figure(figsize=(4, 3))
plt.imshow(cm_svm, interpolation="nearest", cmap="Oranges")
# plt.title("Confusion matrix – Linear SVM")
plt.colorbar()
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel("Predicted")
plt.ylabel("True")

thresh = cm_svm.max() / 2.0
for i in range(cm_svm.shape[0]):
    for j in range(cm_svm.shape[1]):
        plt.text(
            j,
            i,
            f"{cm_svm[i, j]:.2f}",
            ha="center",
            va="center",
            color="white" if cm_svm[i, j] > thresh else "black",
        )

plt.tight_layout()
plt.savefig("static/metrics/bias_confusion_svm.png", bbox_inches="tight")
plt.close()


# ----------------- 3) Simple model-comparison bar chart -----------------

macro_f1_lr = f1_score(y_test, y_pred_lr, average="macro")
# macro_f1_svm = f1_score(y_test, y_pred_svm, average="macro")
macro_f1_teacher = f1_score(y_test, y_pred_teacher, average="macro")

acc_lr = accuracy_score(y_test, y_pred_lr)
# acc_svm = accuracy_score(y_test, y_pred_svm)
acc_teacher = accuracy_score(y_test, y_pred_teacher)

models = ["Logistic Regression", "BERTweet baseline"]
macro_f1_vals = [macro_f1_lr, macro_f1_teacher]
acc_vals = [acc_lr, acc_teacher]


plt.figure(figsize=(5, 3))
x = np.arange(len(models))
width = 0.35
plt.bar(x - width/2, macro_f1_vals, width, label="Macro F1")
plt.bar(x + width/2, acc_vals, width, label="Accuracy")
plt.xticks(x, models)
plt.ylim(0, 1.0)
plt.ylabel("Score")
plt.title("Model comparison (LR vs BERTweet baseline)")
plt.legend()
plt.tight_layout()
plt.savefig("static/metrics/model_comparison.png", bbox_inches="tight")
plt.close()

print("\nSaved LR + BERTweet reports and plots to static/metrics/")
