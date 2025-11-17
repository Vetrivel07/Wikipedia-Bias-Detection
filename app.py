import os
import io
import base64
from flask import Flask, render_template, request, session
import joblib
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# plotting and word cloud
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Ensure VADER is available
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")


BIAS_THRESHOLD = float(os.environ.get("BIAS_THRESHOLD", 0.40))

MODEL_DIR = os.environ.get("BIAS_MODEL_DIR", "models")
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
MODEL_PATH = os.path.join(MODEL_DIR, "bias_lr_calibrated.joblib")

tfidf = joblib.load(TFIDF_PATH)
model = joblib.load(MODEL_PATH)
vader = SentimentIntensityAnalyzer()

app = Flask(__name__)
app.secret_key = "secret" 

# ---------- helper functions for charts ----------

def fig_to_base64():
    """Convert current matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_bytes = buf.getvalue()
    buf.close()
    return base64.b64encode(img_bytes).decode("ascii")


def make_bias_chart(bias_probas_pct: dict) -> str:
    """
    bias_probas_pct: dict like {'NEGATIVE': 12.8, 'NEUTRAL': 82.8, 'POSITIVE': 4.3}
    Returns base64 PNG of a small bar chart.
    """
    plt.figure(figsize=(3.5, 3))
    labels = ["Neutral", "Positive Bias", "Negative Bias"]
    values = [
        bias_probas_pct.get("NEUTRAL", 0.0),
        bias_probas_pct.get("POSITIVE", 0.0),
        bias_probas_pct.get("NEGATIVE", 0.0),
    ]
    plt.bar(labels, values)
    plt.ylabel("Probability (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=20)
    # plt.title("Bias probabilities")
    img = fig_to_base64()
    plt.close()
    return img


def make_sentiment_chart(sent_pct: dict) -> str:
    """
    sent_pct: dict like {'neg': 6.6, 'neu': 80.4, 'pos': 13.1}
    Returns base64 PNG of a small bar chart.
    """
    plt.figure(figsize=(3.5, 3))
    labels = ["Negative", "Neutral", "Positive"]
    values = [
        sent_pct.get("neg", 0.0),
        sent_pct.get("neu", 0.0),
        sent_pct.get("pos", 0.0),
    ]
    plt.bar(labels, values)
    plt.ylabel("Probability (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=20)
    # plt.title("Sentiment probabilities")
    img = fig_to_base64()
    plt.close()
    return img


def make_wordcloud(text: str) -> str:
    """Generate a word cloud PNG (base64) for the given article text."""
    # You can add stopwords if you want to clean further
    wc = WordCloud(width=600, height=300, background_color="white").generate(text)
    plt.figure(figsize=(6, 3))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word cloud")
    img = fig_to_base64()
    plt.close()
    return img


# ---------- existing bias + sentiment functions ----------

def get_bias(text: str):
    X = tfidf.transform([text])
    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)

    # main bias label
    pred_idx = proba.argmax()
    label = classes[pred_idx]

    # per-class probabilities in %
    class_probas_pct = {
        cls: round(float(p) * 100.0, 1)
        for cls, p in zip(classes, proba)
    }

    p_neg = float(proba[classes.index("NEGATIVE")]) if "NEGATIVE" in classes else 0.0
    p_pos = float(proba[classes.index("POSITIVE")]) if "POSITIVE" in classes else 0.0

    bias_strength_pct = round((p_neg + p_pos) * 100.0, 1)

    # IMPORTANT: cast to Python bool
    is_biased = bool((p_neg + p_pos) >= BIAS_THRESHOLD)

    return label, class_probas_pct, bias_strength_pct, is_biased, classes


def get_sentiment(text: str):
    s = vader.polarity_scores(text)
    pct = {
        "neg": round(s["neg"] * 100.0, 1),
        "neu": round(s["neu"] * 100.0, 1),
        "pos": round(s["pos"] * 100.0, 1),
    }

    # sentiment label
    if s["pos"] >= s["neg"] and s["pos"] >= s["neu"]:
        label = "Positive"
    elif s["neg"] >= s["pos"] and s["neg"] >= s["neu"]:
        label = "Negative"
    else:
        label = "Neutral"

    return label, pct, round(s["compound"], 4)

def get_bias_paragraph(text: str):
    """
    Run bias detection paragraph-wise and average class probabilities.
    Also returns per-paragraph details for UI.
    Returns:
      label, class_probas_pct, bias_strength_pct, is_biased, classes, paragraph_details
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        label, pct, strength, is_biased, classes = get_bias(text)
        return label, pct, strength, is_biased, classes, []

    sum_probas = None
    classes = None
    paragraph_details = []

    for idx, para in enumerate(paragraphs, start=1):
        para_label, class_pct, para_strength, para_is_biased, cls_list = get_bias(para)

        if classes is None:
            classes = cls_list

        vec = [class_pct[c] for c in classes]
        if sum_probas is None:
            sum_probas = vec
        else:
            sum_probas = [a + b for a, b in zip(sum_probas, vec)]

        # store detailed info for this paragraph
        preview = para[:220] + ("..." if len(para) > 220 else "")
        paragraph_details.append(
            {
                "index": idx,
                "label": para_label,
                "class_probas_pct": class_pct,
                "bias_strength_pct": para_strength,
                "is_biased": para_is_biased,
                "preview": preview,
            }
        )

    avg = [round(v / len(paragraphs), 1) for v in sum_probas]
    class_probas_pct = {cls: avg[i] for i, cls in enumerate(classes)}

    label = max(class_probas_pct, key=class_probas_pct.get)

    p_neg = class_probas_pct.get("NEGATIVE", 0.0)
    p_pos = class_probas_pct.get("POSITIVE", 0.0)
    bias_strength_pct = round(p_neg + p_pos, 1)

    is_biased = bool((p_neg + p_pos) / 100.0 >= BIAS_THRESHOLD)

    return label, class_probas_pct, bias_strength_pct, is_biased, classes, paragraph_details


# ---------- Flask route ----------

@app.route("/", methods=["GET", "POST"])
def home():
    ctx = {
        "text": "",
        "result": None,
        "threshold": BIAS_THRESHOLD,
        "classes": None,
        "bias_plot": None,
        "para_plot": None,
        "sent_plot": None,
        "wordcloud": None,
        "paragraph_details": None,
    }

    if request.method == "POST":
        action = request.form.get("action", "analyze")

        if action == "clear":
            session.pop("last_text", None)
            return render_template("home.html", **ctx)

        text = (request.form.get("text") or "").strip()

        if action == "analyze" and not text:
            ctx["error"] = "Please paste your article before analyzing."
            return render_template("home.html", **ctx)

        # Save last input to session
        session["last_text"] = text

         # --- minimum words check for processing ---
        MIN_WORDS = 300
        if action == "analyze" and len(text.split()) < MIN_WORDS:
            ctx["error"] = "Please paste an article with at least 300 words for reliable bias detection."
            ctx["text"] = text
            return render_template("home.html", **ctx)

        if text:
            # analysis...
            bias_label, bias_pct, bias_strength, is_biased, classes = get_bias(text)
            sent_label, sent_pct, sent_compound = get_sentiment(text)
            (para_label, para_pct, para_strength, para_is_biased, _, para_details) = get_bias_paragraph(text)

            ctx.update({
                "text": text,
                "result": {
                    "bias": {
                        "label": bias_label,
                        "class_probas_pct": bias_pct,
                        "bias_strength_pct": bias_strength,
                        "is_biased": is_biased,
                    },
                    "paragraph_bias": {
                        "label": para_label,
                        "class_probas_pct": para_pct,
                        "bias_strength_pct": para_strength,
                        "is_biased": para_is_biased,
                    },
                    "sentiment": {
                        "label": sent_label,
                        "percentages": sent_pct,
                        "compound": sent_compound,
                    },
                },
                "classes": classes,
                "bias_plot": make_bias_chart(bias_pct),
                "para_plot": make_bias_chart(para_pct),
                "sent_plot": make_sentiment_chart(sent_pct),
                "wordcloud": make_wordcloud(text),
                "paragraph_details": para_details,
            })

    else:
        ctx["text"] = session.get("last_text", "")

    return render_template("home.html", **ctx)



@app.route("/metrics")
def metrics():
    lr_report = None
    svm_report = None
    teacher_report = None

    lr_path = os.path.join("static", "metrics", "lr_report.txt")
    svm_path = os.path.join("static", "metrics", "svm_report.txt")
    teacher_path = os.path.join("static", "metrics", "teacher_report.txt")

    if os.path.exists(lr_path):
        with open(lr_path, "r", encoding="utf-8") as f:
            lr_report = f.read()

    if os.path.exists(svm_path):
        with open(svm_path, "r", encoding="utf-8") as f:
            svm_report = f.read()

    if os.path.exists(teacher_path):
        with open(teacher_path, "r", encoding="utf-8") as f:
            teacher_report = f.read()

    return render_template(
        "metrics.html",
        lr_report=lr_report,
        svm_report=svm_report,
        teacher_report=teacher_report,
    )


if __name__ == "__main__":
    app.run(debug=True)
