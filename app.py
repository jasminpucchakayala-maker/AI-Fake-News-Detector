import csv
import io
import os
import sqlite3
from datetime import datetime
from functools import wraps

from dotenv import load_dotenv
from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from werkzeug.security import check_password_hash, generate_password_hash

from news_api import fetch_latest_headlines
from assistant import generate_ai_assistant_response
from analytics import build_dashboard_payload
from model_utils import (
    evaluate_source_credibility,
    load_or_train_models,
    preprocess_text,
    predict_news,
    retrain_models_from_dataset,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "app.db")

load_dotenv(os.path.join(BASE_DIR, ".env"))

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-in-production")
app.config["NEWS_API_KEY"] = os.getenv("NEWS_API_KEY", "")


def get_db_connection():
    """Create SQLite connection with dict-like row access."""
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    """Create all required tables if they do not exist."""
    with get_db_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                is_admin INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                article_text TEXT NOT NULL,
                source_url TEXT,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                source_label TEXT NOT NULL,
                source_score INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS saved_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                prediction_id INTEGER NOT NULL,
                report_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (prediction_id) REFERENCES predictions (id)
            )
            """
        )
        connection.commit()

        # Create an admin user for quick testing.
        existing_admin = cursor.execute("SELECT id FROM users WHERE email = ?", ("admin@demo.com",)).fetchone()
        if not existing_admin:
            cursor.execute(
                """
                INSERT INTO users (name, email, password_hash, is_admin, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "Administrator",
                    "admin@demo.com",
                    generate_password_hash("admin123"),
                    1,
                    datetime.utcnow().isoformat(),
                ),
            )
            connection.commit()


def login_required(view_function):
    """Protect pages that require an authenticated user."""

    @wraps(view_function)
    def wrapped_view(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return view_function(*args, **kwargs)

    return wrapped_view


def admin_required(view_function):
    """Protect pages accessible only by admins."""

    @wraps(view_function)
    def wrapped_view(*args, **kwargs):
        if not session.get("is_admin"):
            flash("Admin access is required.", "danger")
            return redirect(url_for("dashboard"))
        return view_function(*args, **kwargs)

    return wrapped_view


@app.context_processor
def inject_user():
    """Expose logged-in user information in all templates."""
    return {
        "current_user_name": session.get("user_name"),
        "is_admin": bool(session.get("is_admin")),
    }


@app.route("/")
def landing():
    """Render the public landing page."""
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    """Handle user registration."""
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()

        if not name or not email or not password:
            flash("All fields are required.", "danger")
            return redirect(url_for("register"))
        if len(password) < 6:
            flash("Password must be at least 6 characters long.", "danger")
            return redirect(url_for("register"))

        with get_db_connection() as connection:
            cursor = connection.cursor()
            existing_user = cursor.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
            if existing_user:
                flash("Email is already registered.", "warning")
                return redirect(url_for("register"))
            cursor.execute(
                """
                INSERT INTO users (name, email, password_hash, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (name, email, generate_password_hash(password), datetime.utcnow().isoformat()),
            )
            connection.commit()

        flash("Registration successful. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """Handle user login and session creation."""
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()

        with get_db_connection() as connection:
            user = connection.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            session["is_admin"] = user["is_admin"]
            flash("Welcome back!", "success")
            return redirect(url_for("dashboard"))

        flash("Invalid email or password.", "danger")
        return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    """Clear session and log out user."""
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("landing"))


@app.route("/dashboard")
@login_required
def dashboard():
    """Render dashboard with analytics and history summary."""
    payload = build_dashboard_payload(get_db_connection, session["user_id"])
    return render_template("dashboard.html", payload=payload)


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    """Predict fake/real label for submitted article text."""
    article_text = request.form.get("article_text", "").strip()
    source_url = request.form.get("source_url", "").strip()

    if len(article_text) < 30:
        flash("Article text is too short. Please provide more details.", "danger")
        return redirect(url_for("dashboard"))

    cleaned_text = preprocess_text(article_text)
    prediction_label, confidence_score = predict_news(cleaned_text)
    source_label, source_score = evaluate_source_credibility(source_url)

    with get_db_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO predictions (
                user_id, article_text, source_url, prediction, confidence,
                source_label, source_score, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session["user_id"],
                article_text,
                source_url,
                prediction_label,
                confidence_score,
                source_label,
                source_score,
                datetime.utcnow().isoformat(),
            ),
        )
        prediction_id = cursor.lastrowid
        connection.commit()

    flash(
        f"Prediction: {prediction_label} ({confidence_score:.2f}% confidence), Source: {source_label}.",
        "success",
    )
    return redirect(url_for("prediction_result", prediction_id=prediction_id))


@app.route("/prediction/<int:prediction_id>")
@login_required
def prediction_result(prediction_id):
    """Show details for a single prediction."""
    with get_db_connection() as connection:
        prediction = connection.execute(
            "SELECT * FROM predictions WHERE id = ? AND user_id = ?",
            (prediction_id, session["user_id"]),
        ).fetchone()
    if not prediction:
        flash("Prediction not found.", "warning")
        return redirect(url_for("history"))
    return render_template("prediction_result.html", prediction=prediction)


@app.route("/history")
@login_required
def history():
    """Render searchable prediction history."""
    query = request.args.get("q", "").strip()
    sql = "SELECT * FROM predictions WHERE user_id = ?"
    params = [session["user_id"]]
    if query:
        sql += " AND (article_text LIKE ? OR prediction LIKE ? OR source_url LIKE ?)"
        wildcard = f"%{query}%"
        params.extend([wildcard, wildcard, wildcard])
    sql += " ORDER BY created_at DESC"

    with get_db_connection() as connection:
        rows = connection.execute(sql, params).fetchall()
    return render_template("history.html", rows=rows, query=query)


@app.route("/save-report/<int:prediction_id>", methods=["POST"])
@login_required
def save_report(prediction_id):
    """Save prediction record as a named report."""
    report_name = request.form.get("report_name", "").strip() or f"Report-{prediction_id}"
    with get_db_connection() as connection:
        prediction = connection.execute(
            "SELECT id FROM predictions WHERE id = ? AND user_id = ?",
            (prediction_id, session["user_id"]),
        ).fetchone()
        if not prediction:
            flash("Prediction not found for saving report.", "danger")
            return redirect(url_for("history"))

        connection.execute(
            """
            INSERT INTO saved_reports (user_id, prediction_id, report_name, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (session["user_id"], prediction_id, report_name, datetime.utcnow().isoformat()),
        )
        connection.commit()

    flash("Report saved successfully.", "success")
    return redirect(url_for("history"))


@app.route("/reports/export/csv")
@login_required
def export_csv():
    """Export user predictions as CSV."""
    with get_db_connection() as connection:
        rows = connection.execute(
            """
            SELECT article_text, source_url, prediction, confidence, source_label, source_score, created_at
            FROM predictions
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (session["user_id"],),
        ).fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Article", "Source URL", "Prediction", "Confidence", "Source Label", "Source Score", "Timestamp"])
    for row in rows:
        writer.writerow(
            [
                row["article_text"],
                row["source_url"],
                row["prediction"],
                f"{row['confidence']:.2f}",
                row["source_label"],
                row["source_score"],
                row["created_at"],
            ]
        )
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="prediction_reports.csv",
    )


@app.route("/reports/export/pdf")
@login_required
def export_pdf():
    """Export user predictions as PDF."""
    with get_db_connection() as connection:
        rows = connection.execute(
            """
            SELECT prediction, confidence, source_label, created_at
            FROM predictions
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT 30
            """,
            (session["user_id"],),
        ).fetchall()

    memory = io.BytesIO()
    pdf = canvas.Canvas(memory, pagesize=letter)
    pdf.setTitle("Fake News Detection Report")
    pdf.drawString(50, 760, "Fake News Detector - User Report")
    pdf.drawString(50, 745, f"Generated at: {datetime.utcnow().isoformat()}")
    y_axis = 720
    for row in rows:
        line = f"{row['created_at'][:19]} | {row['prediction']} | {row['confidence']:.2f}% | {row['source_label']}"
        pdf.drawString(50, y_axis, line[:110])
        y_axis -= 20
        if y_axis <= 50:
            pdf.showPage()
            y_axis = 760
    pdf.save()
    memory.seek(0)
    return send_file(memory, mimetype="application/pdf", as_attachment=True, download_name="prediction_reports.pdf")


@app.route("/live-news")
@login_required
def live_news():
    """Fetch latest headlines from NewsAPI and show results."""
    headlines = fetch_latest_headlines(app.config["NEWS_API_KEY"])
    return render_template("live_news.html", headlines=headlines)


@app.route("/chatbot", methods=["POST"])
@login_required
def chatbot():
    """AI assistant endpoint for explanation and Q&A."""
    article_text = request.form.get("article_text", "").strip()
    user_question = request.form.get("user_question", "").strip()
    response = generate_ai_assistant_response(article_text, user_question)
    return jsonify({"response": response})


@app.route("/admin")
@login_required
@admin_required
def admin_panel():
    """Admin panel for user and report management."""
    with get_db_connection() as connection:
        users = connection.execute(
            "SELECT id, name, email, is_admin, created_at FROM users ORDER BY created_at DESC"
        ).fetchall()
        reports = connection.execute(
            """
            SELECT p.id, u.email, p.prediction, p.confidence, p.created_at
            FROM predictions p
            JOIN users u ON u.id = p.user_id
            ORDER BY p.created_at DESC
            LIMIT 100
            """
        ).fetchall()
    return render_template("admin.html", users=users, reports=reports)


@app.route("/admin/delete-report/<int:prediction_id>", methods=["POST"])
@login_required
@admin_required
def delete_report(prediction_id):
    """Allow admin to delete suspicious/incorrect report entries."""
    with get_db_connection() as connection:
        connection.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
        connection.commit()
    flash("Report deleted successfully.", "success")
    return redirect(url_for("admin_panel"))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint for prediction."""
    payload = request.get_json(silent=True) or {}
    article_text = payload.get("article_text", "").strip()
    source_url = payload.get("source_url", "").strip()

    if len(article_text) < 30:
        return jsonify({"error": "article_text must contain at least 30 characters"}), 400

    cleaned_text = preprocess_text(article_text)
    prediction_label, confidence_score = predict_news(cleaned_text)
    source_label, source_score = evaluate_source_credibility(source_url)
    return jsonify(
        {
            "prediction": prediction_label,
            "confidence": round(confidence_score, 2),
            "source_label": source_label,
            "source_score": source_score,
        }
    )


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    """Retrain the traditional ML model from a provided CSV dataset path."""
    payload = request.get_json(silent=True) or {}
    dataset_path = payload.get("dataset_path", "").strip() or os.path.join(BASE_DIR, "sample_news.csv")
    try:
        metrics = retrain_models_from_dataset(dataset_path)
        return jsonify({"message": "Retraining complete", "metrics": metrics})
    except Exception as error:
        return jsonify({"error": str(error)}), 400


@app.errorhandler(404)
def not_found(_error):
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_error(_error):
    return render_template("500.html"), 500


init_db()
load_or_train_models()


if __name__ == "__main__":
    app.run(debug=True)
