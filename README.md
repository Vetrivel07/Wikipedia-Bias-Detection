# Wikipedia Bias Detection — Flask + NLP Machine Learning Web Application

## 1. Technology Stack

This project is implemented using the following technologies:

- **Flask (Python)** — selected for lightweight routing, rapid prototyping, and direct support for server-side HTML rendering with Jinja.
- **Scikit-learn (ML)** — used to train the primary bias detection model using TF-IDF vectorization and Logistic Regression with probability calibration.
- **VADER Sentiment Analyzer (NLTK)** — chosen for rule-based sentiment classification optimized for social and general English text.
- **Matplotlib** — used for generating dynamic bar-chart visualizations of model outputs.
- **Jinja2 Templates + Custom CSS** — providing a clean, production-style UI without the complexity of a frontend framework.
- **OpenStreetMap Nominatim API** — used for address-to-coordinate conversion in live user geospatial queries.

This stack was selected to prioritize fast inference, lightweight deployment, interpretable NLP outputs, and complete transparency over model decisions, while keeping the system easy to extend toward transformer-based models later.

---

## 2. Process (Data Loading Workflow)
 - The system operates using a structured pipeline for both offline model training and online inference inside the Flask application.
 - first built a dataset consisting of Wikipedia article excerpts, which were weak-labeled using a Hugging Face `finiteautomata/bertweet-base-sentiment-analysis` model for initial polarity signals.
---

## 3. Web Application Workflow

- Users paste a Wikipedia article or long paragraph into the text box.
- A minimum length check (300 words) ensures that bias predictions operate on sufficiently large samples.
- The Flask backend then:

    1. Runs full-text bias detection
    2. Runs paragraph-level bias detection with per-paragraph summaries
    3. Computes sentiment classification via VADER
    4. Generates bias charts, sentiment charts, and word clouds
    5. Results are dynamically rendered using Jinja templates and inline base64 images.

**All intermediate inference artifacts (plots, probabilities, paragraph metadata) are generated on-demand for each request.**

---

## 3. Volume 

 - The system has been tested with realistic Wikipedia-style datasets.
- Typical training artefact counts: Dataset size (training): ~10,000 article excerpts
- TF-IDF vocabulary size: ~50,000 terms
- Classes: `NEGATIVE`, `NEUTRAL`, `POSITIVE`
---

## 4. Variety (Search Scenarios Demonstrated)

This application supports multiple informative and interactive analysis modes:

- Full-text bias detection:
Returns the dominant bias class along with weighted class probabilities.
- Bias-strength scoring:
Measures combined positive + negative bias probability as a % indicator.
- Paragraph-level analysis:
Automatically splits article into meaningful segments and highlights:
    - per-paragraph bias label
    - per-paragraph bias probability distribution
    - local bias strength
    - preview text for explanation
- Sentiment analysis:
Determines emotional tone using VADER’s negative/neutral/positive scoring.
- Word cloud visualization:
Extracts and displays top keywords from the article to show topic distribution.

These features provide users with a comprehensive understanding of both bias direction and linguistic tone, closely reflecting real-world media analysis tools.
---

## 6. Model Reliability & Background

This project does not perform blind classification. Measures have been taken to ensure reliability:

- The deployed model uses probability-calibrated LR, not raw SVM scores.
- Inputs under 300 words are rejected for reliability.
- The teacher model (BERTweet) is included only for transparency and comparison.
- All confusion matrices and reports are openly available.
- The threshold for bias detection is environment-configurable (BIAS_THRESHOLD).

This mirrors production-oriented ML deployments where traceability, explainability, and probability stability are essential.

---

## 6. How to Run Locally (Flask + MongoDB)

**Requirements:**  
- Python 3.8+  
- MongoDB running locally (default port `27017`)  
- All model files present in `/models`


### Step 1 — Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate      # Windows

```

### Step 2 — Install required dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Start the Flask server
```bash
python app.py
```

### Step 4 — Open the application in your browser
```
http://localhost:3000
```
You can now:

- Paste a full Wikipedia article
- View bias classification
- Explore paragraph-level insights
- Check sentiment
- View word cloud
- Inspect model evaluation metrics
---

## 7. Preview

### 🏠 Home Page
![Index Page](static/Preview/img1.png)

---

## 8. Bells and Whistles 
This project extends significantly beyond a simple “text classifier” and demonstrates real-world NLP product design.

Some standout features:

1. Paragraph-Level Bias Intelligence
Not just classifying whole articles — the system isolates localized bias hotspots.

2. Probability Calibration for Stability
Ensures prediction scores remain interpretable and reliable.

3. Charted Probability Visualizations
Inline Matplotlib images makes outputs polished and professional.

4. Live Word Cloud Generation
Highlights topical emphasis and linguistic patterns on demand.

5. Transparent Metrics Page
Shows full classification reports & confusion matrices for academic rigor.

6. User-Friendly Web UI
Custom navbar, tooltips, error messages, session persistence, and inline content help bridge ML and user experience seamlessly.

7. Future-Ready Architecture
I designed the architecture so the system can later support paste-a-link analysis, where users enter a Wikipedia URL and the app automatically retrieves and analyzes the article text. This will use the Wikipedia REST API instead of heavy HTML scraping to ensure fast, consistent, and secure extraction. The backend already has the scaffolding for this feature; it will be enabled in a future version.

---

## 9. Conclusion
This project demonstrates a production-aware **NLP application** with strong emphasis on interpretability, usability, and responsible ML deployment. By integrating calibrated **machine learning**, **sentiment analysis**, **paragraph-level diagnostics**, and **dynamic visualization**, it provides a rounded, deeply informative analysis workflow ideal for evaluating potential bias in Wikipedia-style content.
The system is deliberately engineered to be modular and extensible — supporting future upgrades to **transformer-based classifiers**, **dataset expansion**, **additional visualization tools**, or integration into larger **fact-checking pipelines**.