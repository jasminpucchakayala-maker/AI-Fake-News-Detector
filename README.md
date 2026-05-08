# AI Fake News Detector (Flask + Scikit-learn + SQLite)

A full-stack, resume-ready AI Fake News Detection platform with:

- User authentication (register/login/logout, password hashing, session management)
- ML-powered FAKE/REAL prediction with confidence score
- Optional deep learning model training support (TensorFlow)
- Source credibility verification score
- Prediction history with search
- CSV/PDF report export
- Dashboard analytics with Chart.js
- Admin panel for user/report management
- Live News API integration
- AI assistant chatbot for explanation, summary, and misinformation guidance
- REST API endpoints for prediction and retraining

---

## 1) Project Structure

```text
fake-news/
├── app.py
├── train_model.py
├── requirements.txt
├── Procfile
├── runtime.txt
├── render.yaml
├── netlify.toml
├── .env.example
├── README.md
├── api/
│   ├── __init__.py
│   └── news_api.py
├── chatbot/
│   ├── __init__.py
│   └── assistant.py
├── dashboard/
│   ├── __init__.py
│   └── analytics.py
├── models/
│   ├── __init__.py
│   └── model_utils.py
├── database/
│   └── README.md
├── data/
│   └── sample_news.csv
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── prediction_result.html
│   ├── history.html
│   ├── live_news.html
│   ├── admin.html
│   ├── 404.html
│   └── 500.html
└── static/
    ├── css/
    │   └── styles.css
    └── js/
        └── main.js
```

---

## 2) Setup Instructions (Local)

### Step 1: Clone repository

```bash
git clone <your-repo-url>
cd fake-news
```

### Step 2: Create virtual environment

```bash
python -m venv venv
```

Activate:

- Windows PowerShell: `venv\Scripts\Activate.ps1`
- Linux/macOS: `source venv/bin/activate`

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure environment variables

Copy `.env.example` to `.env` and update values:

```env
FLASK_SECRET_KEY=replace-with-strong-secret
NEWS_API_KEY=your-newsapi-key-here
```

### Step 5: Train model (optional manual run)

```bash
python train_model.py --dataset data/sample_news.csv
```

### Step 6: Run application

```bash
python app.py
```

Open: [http://127.0.0.1:5000](http://127.0.0.1:5000)

### Step 7: Run unit tests

```bash
pytest -q
```

---

## 3) Default Admin Login

- Email: `admin@demo.com`
- Password: `admin123`

---

## 4) REST API Endpoints

- `POST /api/predict`
  - JSON body:
    ```json
    {
      "article_text": "Your article content...",
      "source_url": "https://example.com/news"
    }
    ```

- `POST /api/retrain`
  - JSON body:
    ```json
    {
      "dataset_path": "data/sample_news.csv"
    }
    ```

---

## 5) Model Pipeline Details

- Preprocessing: lowercase, URL removal, punctuation removal, whitespace normalization
- Vectorizer: TF-IDF (`max_features=6000`, bi-grams)
- Classifier: Logistic Regression
- Metrics: accuracy + classification report
- Model persistence: `joblib`
- Optional deep learning: TensorFlow simple neural model stored in `models/`

---

## 6) Deployment

### Docker (One-command deployment)

1. Create `.env` from `.env.example`.
2. Build and run:

```bash
docker compose up --build
```

3. Open: [http://127.0.0.1:5000](http://127.0.0.1:5000)

### Render (Recommended for Flask backend)

1. Push code to GitHub.
2. Create new Web Service in Render.
3. Use:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app`
4. Add environment variables:
   - `FLASK_SECRET_KEY`
   - `NEWS_API_KEY`
5. Deploy.

### Netlify (Frontend shell + API proxy)

Use `netlify.toml` to deploy static frontend previews and proxy `/api/*` routes to your Render backend URL.

---

## 7) GitHub Upload Steps

```bash
git init
git add .
git commit -m "Build full-stack AI fake news detector platform"
git branch -M main
git remote add origin <your-repository-url>
git push -u origin main
```

---

## 8) Resume Highlights

- Built full-stack AI misinformation detection platform with real-time predictions.
- Implemented secure auth, role-based admin panel, report export, and analytics dashboard.
- Designed modular ML training pipeline with retraining support and deploy-ready architecture.
