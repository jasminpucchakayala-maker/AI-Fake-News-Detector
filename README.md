 AI Fake News Detector (Flask + Scikit-learn + SQLite)

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

 1) Project Structure

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
