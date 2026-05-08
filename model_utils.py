import os
import re
import string
from urllib.parse import urlparse

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fake_news_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.txt")
SAMPLE_DATASET_PATH = os.path.join(BASE_DIR, "sample_news.csv")

model_pipeline = None


def preprocess_text(text):
    """Normalize text for better model quality."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ensure_sample_dataset_exists():
    """Create a starter dataset if missing so beginners can run immediately."""
    if os.path.exists(SAMPLE_DATASET_PATH):
        return
    rows = [
        {"text": "Government confirms new healthcare reforms after parliament vote.", "label": "REAL"},
        {"text": "Scientists discover new exoplanet using advanced telescope data.", "label": "REAL"},
        {"text": "Breaking: Celebrity says drinking bleach cures all diseases instantly.", "label": "FAKE"},
        {"text": "Fake report claims moon emits healing frequencies every midnight.", "label": "FAKE"},
        {"text": "Central bank releases official inflation report for this quarter.", "label": "REAL"},
        {"text": "Conspiracy post says internet will shut down forever tomorrow morning.", "label": "FAKE"},
    ]
    pd.DataFrame(rows).to_csv(SAMPLE_DATASET_PATH, index=False)


def train_classic_model(dataset_path):
    """Train a TF-IDF + Logistic Regression pipeline and persist it."""
    dataframe = pd.read_csv(dataset_path)
    if "text" not in dataframe.columns or "label" not in dataframe.columns:
        raise ValueError("Dataset must include 'text' and 'label' columns.")

    dataframe["text"] = dataframe["text"].fillna("").map(preprocess_text)
    dataframe["label"] = dataframe["label"].str.upper().str.strip()
    dataframe = dataframe[dataframe["label"].isin(["FAKE", "REAL"])]
    if len(dataframe) < 4:
        raise ValueError("Dataset is too small. Add more rows for training.")

    train_x, test_x, train_y, test_y = train_test_split(
        dataframe["text"], dataframe["label"], test_size=0.25, random_state=42, stratify=dataframe["label"]
    )

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=6000, ngram_range=(1, 2))),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(train_x, train_y)

    predictions = pipeline.predict(test_x)
    accuracy = accuracy_score(test_y, predictions)
    report = classification_report(test_y, predictions)

    joblib.dump(pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as metrics_file:
        metrics_file.write(f"Accuracy: {accuracy:.4f}\n\n{report}\n")

    # Optional deep learning support for advanced users.
    train_deep_learning_model(dataframe)

    return {"accuracy": round(float(accuracy), 4), "report": report}


def train_deep_learning_model(dataframe):
    """Optional TensorFlow LSTM training (safe fallback when TF is not installed)."""
    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.preprocessing.text import Tokenizer
    except Exception:
        return {"enabled": False, "reason": "TensorFlow not installed"}

    texts = dataframe["text"].astype(str).tolist()
    labels = [1 if value == "REAL" else 0 for value in dataframe["label"].tolist()]
    tokenizer = Tokenizer(num_words=6000, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=120, padding="post")

    model = Sequential(
        [
            Embedding(input_dim=6000, output_dim=32),
            GlobalAveragePooling1D(),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(padded, labels, epochs=3, batch_size=8, verbose=0)

    tf_model_path = os.path.join(BASE_DIR, "deep_learning_model.keras")
    token_path = os.path.join(BASE_DIR, "deep_learning_tokenizer.joblib")
    model.save(tf_model_path)
    joblib.dump(tokenizer, token_path)
    return {"enabled": True, "model_path": tf_model_path}


def load_or_train_models():
    """Load existing model or train from starter dataset."""
    global model_pipeline
    ensure_sample_dataset_exists()
    if not os.path.exists(MODEL_PATH):
        train_classic_model(SAMPLE_DATASET_PATH)
    model_pipeline = joblib.load(MODEL_PATH)
    return model_pipeline


def predict_news(article_text):
    """Return label and confidence percentage."""
    global model_pipeline
    if model_pipeline is None:
        load_or_train_models()
    probabilities = model_pipeline.predict_proba([article_text])[0]
    labels = model_pipeline.classes_
    max_index = int(probabilities.argmax())
    label = labels[max_index]
    confidence = float(probabilities[max_index] * 100)
    return label, confidence


def retrain_models_from_dataset(dataset_path):
    """Public retraining helper used by scripts and API."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    metrics = train_classic_model(dataset_path)
    load_or_train_models()
    return metrics


def evaluate_source_credibility(source_url):
    """Simple news source reputation scoring based on domain."""
    trusted_domains = {
        "reuters.com": 95,
        "bbc.com": 92,
        "nytimes.com": 90,
        "thehindu.com": 88,
        "apnews.com": 94,
    }
    if not source_url:
        return "UNVERIFIED", 50
    domain = urlparse(source_url).netloc.replace("www.", "").lower()
    score = trusted_domains.get(domain, 35)
    label = "TRUSTED" if score >= 75 else "UNTRUSTED"
    return label, score
