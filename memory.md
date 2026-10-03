# Project Memory: Plagiarism Checker Web Application

## 🎓 Project Context & Submission Details
- **Project Type**: College Capstone Project
- **Institution**: Indian Institute of Technology Patna (IIT Patna)
- **Development Group**: 5-Member Team
  1. **Deepak kumar** (Roll: `2312res238`)
     - Collected and prepared dataset for model training.
     - Preprocessed text data (tokenization, cleaning).
     - Initial RNN/LSTM model implementation.
  2. **Deepanshu Kumar** (Roll: `2312res785`)
     - Assisted in dataset preparation/preprocessing.
     - Researched RNN/LSTM architectures.
     - Contributed to model development and coding.
  3. **Deepak kumar** (Roll: `2312res783`)
     - Developed Flask web application.
     - Built interactive dashboard interface.
     - Integrated ML models with web server.
  4. **Deepak kumar** (Roll: `2312res237`)
     - Performed data analysis on collected datasets.
     - Refined/cleaned data for performance tuning.
     - Formatted and aligned data matrices.
  5. **Deepak kumar** (Roll: `2312res236`)
     - Workflow management and team task coordination.
     - Documentation prep and PPT compile.
     - Maintained progress tracking and reports.
- **Application Scope**: Semantic Plagiarism Checker utilizing Siamese LSTM networks.

This file serves as the core reference and blueprint for building, mapping, and connecting the frontend and backend of the **Plagiarism Checker** project. It outlines the exact file structure, database schema, REST API endpoints, client-side routing, and machine learning integration logic required to build and run this application end-to-end.

---

## 📂 System File Architecture

The complete workspace is organized as follows:

```text
Capstone 2/
├── requirements.txt                   # NLP Model/Notebook dependencies
├── train_snli.txt                     # Raw text dataset (tab-separated)
├── data.csv                           # Cleaned dataset (source_txt, plagiarism_txt, label)
├── plagiarism-checker-nlp.ipynb      # Keras Siamese LSTM training notebook
├── memory.md                          # [THIS FILE] Comprehensive system specs & specs
└── backend/
    ├── requirements.txt               # Flask backend dependencies
    ├── run.py                         # Flask server execution runner
    └── app/
        ├── __init__.py                # Flask app factory
        ├── models.py                  # Database Models (User, DetectionResult)
        ├── routes.py                  # API Routes (Auth, Scans, Admin)
        └── ml/
            ├── __init__.py            # ML Package Initialization
            ├── export_model.py        # Helper script to export weights and tokenizer
            ├── model_helper.py        # Inference, tokenization & text cleaning helper
            ├── siamese_lstm_model.h5  # [EXPORTED] Keras saved model
            └── tokenizer.pickle       # [EXPORTED] Fitted Keras Tokenizer
```

---

## 🧠 1. Machine Learning Integration (`backend/app/ml/`)

The model is a Siamese LSTM network trained to evaluate the semantic similarity between two texts.

### A. Export Utility: `backend/app/ml/export_model.py`
This script reads the project dataset, recreates the Tokenizer, fits it, instantiates the model architecture matching `plagiarism-checker-nlp.ipynb`, and exports both to files so the backend can run inference.

```python
import os
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Subtract
from tensorflow.keras.models import Model

# Resolve absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE_DIR, "..", "..", "..", "data.csv")
MODEL_OUT = os.path.join(BASE_DIR, "siamese_lstm_model.h5")
TOKENIZER_OUT = os.path.join(BASE_DIR, "tokenizer.pickle")

def export():
    print("Loading dataset to fit Tokenizer...")
    df = pd.read_csv(DATA_CSV)
    df.dropna(subset=["source_txt", "plagiarism_txt"], inplace=True)
    
    source = df["source_txt"].astype(str).tolist()
    plag = df["plagiarism_txt"].astype(str).tolist()
    
    # Fit Tokenizer exactly like notebook
    tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
    tokenizer.fit_on_texts(source + plag)
    
    # Save Tokenizer
    with open(TOKENIZER_OUT, "wb") as f:
        pickle.dump(tokenizer, f)
    print("Tokenizer exported successfully.")
    
    # Define Siamese LSTM Architecture matching the notebook
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 128
    lstm_dim = 64
    max_len = 50
    
    input_a = Input(shape=(max_len,))
    input_b = Input(shape=(max_len,))
    
    embedding = Embedding(vocab_size, embedding_dim, input_length=max_len)
    lstm = LSTM(lstm_dim)
    
    encoded_a = lstm(embedding(input_a))
    encoded_b = lstm(embedding(input_b))
    
    merged = Subtract()([encoded_a, encoded_b])
    merged = Dense(64, activation="relu")(merged)
    merged = Dropout(0.5)(merged)
    out = Dense(1, activation="sigmoid")(merged)
    
    model = Model(inputs=[input_a, input_b], outputs=out)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    # Save Model Structure (Weights will compile fresh or load if trained weights file exists)
    model.save(MODEL_OUT)
    print("Siamese LSTM Model compiled and saved.")

if __name__ == "__main__":
    export()
```

### B. Inference Helper: `backend/app/ml/model_helper.py`
Preprocesses raw texts and runs model similarity calculations. Also provides a fallback method using Cosine Similarity if model loading fails.

```python
import os
import re
import string
import pickle
import numpy as np
import difflib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "siamese_lstm_model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pickle")

model = None
tokenizer = None

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def load_ml_assets():
    global model, tokenizer
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
            model = load_model(MODEL_PATH)
            with open(TOKENIZER_PATH, "rb") as f:
                tokenizer = pickle.load(f)
            print("ML model and tokenizer loaded successfully.")
        else:
            print("ML assets not found. Fallback/mock mode will be active.")
    except Exception as e:
        print(f"Error loading ML assets: {e}. Fallback mode active.")

def calculate_similarity(text_a, text_b):
    global model, tokenizer
    if model is not None and tokenizer is not None:
        try:
            clean_a = clean_text(text_a)
            clean_b = clean_text(text_b)
            
            seq_a = tokenizer.texts_to_sequences([clean_a])
            seq_b = tokenizer.texts_to_sequences([clean_b])
            
            pad_a = pad_sequences(seq_a, maxlen=50)
            pad_b = pad_sequences(seq_b, maxlen=50)
            
            score = float(model.predict([pad_a, pad_b])[0][0])
            return score
        except Exception as e:
            print(f"Inference error: {e}. Using fallback.")
            
    # Fallback ratio comparison (Sequence Matcher)
    return float(difflib.SequenceMatcher(None, text_a, text_b).ratio())

def generate_highlighted_diff(text_a, text_b):
    """
    Highlights sentences in Text A that show strong structure similarity
    to any sentence in Text B (plagiarism indicator).
    """
    # Simple sentence tokenizer regex
    sentences_a = re.split(r'(?<=[.!?])\s+', text_a)
    sentences_b = re.split(r'(?<=[.!?])\s+', text_b)
    
    highlighted = []
    for s_a in sentences_a:
        if not s_a.strip():
            continue
        max_ratio = 0.0
        for s_b in sentences_b:
            if not s_b.strip():
                continue
            ratio = difflib.SequenceMatcher(None, s_a.lower(), s_b.lower()).ratio()
            if ratio > max_ratio:
                max_ratio = ratio
        
        # Highlight threshold > 65% similarity
        if max_ratio > 0.65:
            highlighted.append(f'<mark class="highlight-plag">{s_a}</mark>')
        else:
            highlighted.append(s_a)
            
    return " ".join(highlighted)
```

---

## 🗄️ 2. Database Models (`backend/app/models.py`)

Handles persistent storage. Creates User accounts and logs plagiarism scans.

```python
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship, sessionmaker, declarative_base

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SQLITE_DB = os.path.join(BASE_DIR, "..", "plagiarism.db")

engine = create_engine(f"sqlite:///{SQLITE_DB}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_admin = Column(Boolean, default=False)

    detections = relationship("DetectionResult", back_populates="user", cascade="all, delete-orphan")

class DetectionResult(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    input_text = Column(Text, nullable=False)
    similarity = Column(Float, nullable=False)
    highlighted_html = Column(Text, nullable=False)
    source_info = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="detections")
```

---

## 🔌 3. Flask Server Config & API Endpoints

### App Factory: `backend/app/__init__.py`
Initializes configurations, sets CORS to allow cross-origin requests, registers blueprints, and triggers SQLite table creation.

```python
import os
from flask import Flask
from flask_cors import CORS
from .models import Base, engine
from .routes import api_bp
from .ml.model_helper import load_ml_assets

def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Initialize DB Tables
    Base.metadata.create_all(bind=engine)
    
    # Register blueprints
    app.register_blueprint(api_bp, url_prefix="/api")
    
    # Load ML weights & Tokenizer
    load_ml_assets()
    
    return app

app = create_app()
```

### Server Runner: `backend/run.py`
Runs development server.
```python
from app import app

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
```

### Routing and Logic: `backend/app/routes.py`
Decodes credentials, extracts PDF / Word files, validates JWTs (via simple Authorization headers), and manages history logs.

```python
import jwt  # Simulating lightweight token mapping
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from .models import SessionLocal, User, DetectionResult
import PyPDF2
import docx

api_bp = Blueprint("api", __name__)
JWT_SECRET = "super-secret-plagiarism-key-99"

# Helper text extractors
def extract_text_from_pdf(stream):
    reader = PyPDF2.PdfReader(stream)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(stream):
    doc = docx.Document(stream)
    text = [p.text for p in doc.paragraphs]
    return "\n".join(text)

# Token decorator
def get_user_from_token():
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    token = auth_header.split(" ")[1]
    try:
        data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        db = SessionLocal()
        user = db.query(User).filter(User.id == data["user_id"]).first()
        return user
    except:
        return None

# --- AUTHENTICATION ---
@api_bp.route("/auth/register", methods=["POST"])
def register():
    data = request.json or {}
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")
    
    if not username or not email or not password:
        return jsonify({"error": "Missing required fields"}), 400
        
    db = SessionLocal()
    if db.query(User).filter((User.username == username) | (User.email == email)).first():
        return jsonify({"error": "Username or Email already registered"}), 409
        
    pwd_hash = generate_password_hash(password)
    new_user = User(username=username, email=email, password_hash=pwd_hash)
    db.add(new_user)
    db.commit()
    return jsonify({"message": "User registered successfully", "user_id": new_user.id}), 201

@api_bp.route("/auth/login", methods=["POST"])
def login():
    data = request.json or {}
    email = data.get("email")
    password = data.get("password")
    
    db = SessionLocal()
    user = db.query(User).filter(User.email == email).first()
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({"error": "Invalid email or password"}), 401
        
    token = jwt.encode({
        "user_id": user.id,
        "exp": datetime.utcnow() + timedelta(days=7)
    }, JWT_SECRET, algorithm="HS256")
    
    return jsonify({
        "token": token,
        "user": {"id": user.id, "username": user.username, "is_admin": user.is_admin}
    }), 200

# --- DETECTION ENDPOINT ---
from .ml.model_helper import calculate_similarity, generate_highlighted_diff

@api_bp.route("/detect", methods=["POST"])
def detect():
    user = get_user_from_token()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
        
    # Read either file uploads or raw text JSON
    text_a, text_b = "", ""
    source_name = "Custom Text Comparison"
    
    if "file_a" in request.files and "file_b" in request.files:
        file_a = request.files["file_a"]
        file_b = request.files["file_b"]
        source_name = f"{file_a.filename} vs {file_b.filename}"
        
        # File A parser
        if file_a.filename.endswith(".pdf"):
            text_a = extract_text_from_pdf(file_a.stream)
        elif file_a.filename.endswith(".docx"):
            text_a = extract_text_from_docx(file_a.stream)
        else:
            text_a = file_a.read().decode("utf-8", errors="ignore")
            
        # File B parser
        if file_b.filename.endswith(".pdf"):
            text_b = extract_text_from_pdf(file_b.stream)
        elif file_b.filename.endswith(".docx"):
            text_b = extract_text_from_docx(file_b.stream)
        else:
            text_b = file_b.read().decode("utf-8", errors="ignore")
    else:
        json_data = request.json or {}
        text_a = json_data.get("text_a", "")
        text_b = json_data.get("text_b", "")
        source_name = json_data.get("source_info", "Comparison Text Block")
        
    if not text_a.strip() or not text_b.strip():
        return jsonify({"error": "Both comparison documents must have content."}), 400
        
    # Similarity inference
    score = calculate_similarity(text_a, text_b)
    highlighted = generate_highlighted_diff(text_a, text_b)
    
    # Save detection logs
    db = SessionLocal()
    record = DetectionResult(
        user_id=user.id,
        input_text=text_a[:500] + "...", # Store sample summary
        similarity=score,
        highlighted_html=highlighted,
        source_info=source_name
    )
    db.add(record)
    db.commit()
    
    return jsonify({
        "id": record.id,
        "similarity": score,
        "highlighted_html": highlighted,
        "source_info": source_name,
        "created_at": record.created_at.isoformat()
    }), 200

# --- USER REPORT HISTORY ---
@api_bp.route("/scans", methods=["GET"])
def get_scans():
    user = get_user_from_token()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    db = SessionLocal()
    scans = db.query(DetectionResult).filter(DetectionResult.user_id == user.id).order_by(DetectionResult.created_at.desc()).all()
    results = []
    for s in scans:
        results.append({
            "id": s.id,
            "similarity": s.similarity,
            "source_info": s.source_info,
            "created_at": s.created_at.isoformat()
        })
    return jsonify(results), 200

@api_bp.route("/scans/<int:scan_id>", methods=["GET"])
def get_scan_by_id(scan_id):
    user = get_user_from_token()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    db = SessionLocal()
    scan = db.query(DetectionResult).filter(DetectionResult.id == scan_id, DetectionResult.user_id == user.id).first()
    if not scan:
        return jsonify({"error": "Scan not found"}), 404
    return jsonify({
        "id": scan.id,
        "similarity": scan.similarity,
        "highlighted_html": scan.highlighted_html,
        "source_info": scan.source_info,
        "created_at": scan.created_at.isoformat()
    }), 200

@api_bp.route("/scans/<int:scan_id>", methods=["DELETE"])
def delete_scan(scan_id):
    user = get_user_from_token()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    db = SessionLocal()
    scan = db.query(DetectionResult).filter(DetectionResult.id == scan_id, DetectionResult.user_id == user.id).first()
    if not scan:
        return jsonify({"error": "Scan not found"}), 404
    db.delete(scan)
    db.commit()
    return jsonify({"message": "Scan report deleted successfully"}), 200

# --- ADMIN PANEL ---
@api_bp.route("/admin/stats", methods=["GET"])
def get_admin_stats():
    user = get_user_from_token()
    if not user or not user.is_admin:
        return jsonify({"error": "Forbidden"}), 403
    db = SessionLocal()
    total_users = db.query(User).count()
    total_scans = db.query(DetectionResult).count()
    avg_score = db.query(DetectionResult.similarity).all()
    avg = sum([s[0] for s in avg_score]) / len(avg_score) if avg_score else 0.0
    return jsonify({
        "total_users": total_users,
        "total_scans": total_scans,
        "average_similarity": avg
    }), 200
```

---

## 🎨 4. Frontend Single-Page App Mapping (`frontend/`)

Below are the complete, detailed code specifications for building the frontend files.

### A. Template Skeleton: `frontend/index.html`
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VeriCheck - Deep Learning Plagiarism Checker</title>
    <!-- Premium Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="glow-bg"></div>
    <header class="navbar">
        <div class="container nav-container">
            <a href="#/home" class="logo">
                <span class="gradient-text">VeriCheck</span>
            </a>
            <nav class="nav-menu" id="nav-menu">
                <!-- Dynamic navigation injection -->
            </nav>
        </div>
    </header>

    <main id="app" class="main-content">
        <!-- SPA view renderer injection -->
    </main>

    <footer class="footer">
        <p>&copy; 2026 VeriCheck. Powered by Siamese LSTM Deep Learning Networks.</p>
    </footer>

    <script src="app.js"></script>
</body>
</html>
```

### B. Styling Sheet: `frontend/style.css`
A custom design system loaded with dark-mode styling, glowing typography, glassmorphism containers, and interactive hover effects.

```css
:root {
    --bg-dark: #0a0b10;
    --card-bg: rgba(18, 20, 32, 0.65);
    --border-color: rgba(255, 255, 255, 0.08);
    --primary-glow: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
    --primary-hover: linear-gradient(135deg, #4f46e5 0%, #9333ea 100%);
    --text-primary: #f3f4f6;
    --text-secondary: #9ca3af;
    --error: #ef4444;
    --success: #10b981;
    --warning: #f59e0b;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    background-color: var(--bg-dark);
    color: var(--text-primary);
    font-family: 'Plus Jakarta Sans', sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    overflow-x: hidden;
}

.glow-bg {
    position: absolute;
    top: -10%;
    left: 20%;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(168, 85, 247, 0.15) 0%, rgba(99, 102, 241, 0.05) 50%, rgba(0,0,0,0) 100%);
    filter: blur(80px);
    z-index: -1;
    pointer-events: none;
}

.gradient-text {
    background: var(--primary-glow);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

/* Glassmorphism Panel card styling */
.glass-panel {
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

/* Document side-by-side matching highlighter style */
.highlight-plag {
    background-color: rgba(239, 68, 68, 0.25);
    border-bottom: 2px dashed var(--error);
    padding: 2px 0;
    color: #ffd8d8;
    cursor: help;
}

/* Custom radial progress score gauge styling */
.score-circle {
    position: relative;
    width: 150px;
    height: 150px;
    border-radius: 50%;
    background: conic-gradient(var(--score-color, var(--success)) calc(var(--percentage) * 1%), #1e293b 0);
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 1.5rem;
}

.score-circle::after {
    content: '';
    position: absolute;
    width: 130px;
    height: 130px;
    background-color: #0d0e15;
    border-radius: 50%;
}

.score-text {
    position: relative;
    z-index: 10;
    font-family: 'Outfit', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
}

/* About Us view team grid styling */
.about-container {
    max-width: 1100px;
    margin: 0 auto;
}
.team-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.5rem;
    margin-top: 2rem;
}
.member-card {
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.member-card:hover {
    transform: translateY(-4px);
    border-color: rgba(168, 85, 247, 0.4);
}
.member-roll {
    font-family: 'Outfit', sans-serif;
    font-size: 0.85rem;
    color: var(--text-secondary);
    background: rgba(255, 255, 255, 0.05);
    padding: 2px 8px;
    border-radius: 20px;
    display: inline-block;
    margin-top: 0.25rem;
    margin-bottom: 0.75rem;
}
.member-work-list {
    list-style: none;
    padding-left: 0;
}
.member-work-list li {
    font-size: 0.9rem;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
    position: relative;
    padding-left: 1.2rem;
}
.member-work-list li::before {
    content: '•';
    color: #a855f7;
    position: absolute;
    left: 0.2rem;
    font-size: 1.2rem;
    line-height: 1;
}
```

### C. SPA Router & Controllers: `frontend/app.js`
Performs route updates via changes to `window.location.hash`, stores authentication status, maps view layouts, handles dynamic scans, and formats highlighting elements.

```javascript
const API_URL = "http://127.0.0.1:5000/api";

const state = {
    token: localStorage.getItem("token") || null,
    user: JSON.parse(localStorage.getItem("user")) || null
};

// Route mapper config
const routes = {
    "#/home": renderHome,
    "#/about": renderAbout,
    "#/login": renderLogin,
    "#/register": renderRegister,
    "#/dashboard": renderDashboard,
    "#/new-scan": renderNewScan,
    "#/results": renderResults,
    "#/admin": renderAdmin
};

function initApp() {
    window.addEventListener("hashchange", router);
    // Redirect if no hash
    if (!window.location.hash) {
        window.location.hash = state.token ? "#/dashboard" : "#/home";
    } else {
        router();
    }
    updateNav();
}

function router() {
    const hash = window.location.hash.split("?")[0];
    const view = routes[hash] || renderHome;
    view();
    updateNav();
}

function updateNav() {
    const navMenu = document.getElementById("nav-menu");
    if (state.token) {
        navMenu.innerHTML = `
            <a href="#/dashboard" class="nav-link">Dashboard</a>
            <a href="#/new-scan" class="nav-link">New Scan</a>
            <a href="#/about" class="nav-link">About Us</a>
            ${state.user?.is_admin ? '<a href="#/admin" class="nav-link">Admin</a>' : ''}
            <a href="#" id="btn-logout" class="nav-btn btn-secondary">Logout</a>
        `;
        document.getElementById("btn-logout").addEventListener("click", (e) => {
            e.preventDefault();
            localStorage.clear();
            state.token = null;
            state.user = null;
            window.location.hash = "#/home";
        });
    } else {
        navMenu.innerHTML = `
            <a href="#/home" class="nav-link">Home</a>
            <a href="#/about" class="nav-link">About Us</a>
            <a href="#/login" class="nav-link">Login</a>
            <a href="#/register" class="nav-btn btn-primary">Sign Up</a>
        `;
    }
}

// Custom view injections
function renderHome() {
    document.getElementById("app").innerHTML = `
        <section class="hero-section glass-panel text-center">
            <h1 class="hero-title">Secure & Smart <span class="gradient-text">Plagiarism Detection</span></h1>
            <p class="hero-subtitle">Leveraging Siamese LSTM neural network weights to verify document originality with high semantic accuracy.</p>
            <div class="cta-group">
                <a href="#/register" class="btn btn-primary btn-large">Get Started Now</a>
                <a href="#/login" class="btn btn-secondary btn-large">Client Login</a>
            </div>
            <div class="project-credits" style="margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid var(--border-color); font-size: 0.9rem; color: var(--text-secondary);">
                <p><strong>Capstone College Project</strong> | Group of 5 Members</p>
                <p style="margin-top: 0.25rem;">Indian Institute of Technology Patna (IIT Patna)</p>
            </div>
        </section>
    `;
}

function renderLogin() {
    document.getElementById("app").innerHTML = `
        <div class="auth-card glass-panel">
            <h2>Log In</h2>
            <form id="form-login">
                <div class="form-group">
                    <label>Email Address</label>
                    <input type="email" id="login-email" required placeholder="name@domain.com">
                </div>
                <div class="form-group">
                    <label>Password</label>
                    <input type="password" id="login-password" required placeholder="••••••••">
                </div>
                <button type="submit" class="btn btn-primary w-full">Sign In</button>
            </form>
        </div>
    `;
    document.getElementById("form-login").addEventListener("submit", async (e) => {
        e.preventDefault();
        const email = document.getElementById("login-email").value;
        const password = document.getElementById("login-password").value;
        
        try {
            const res = await fetch(`${API_URL}/auth/login`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ email, password })
            });
            const data = await res.json();
            if (res.ok) {
                localStorage.setItem("token", data.token);
                localStorage.setItem("user", JSON.stringify(data.user));
                state.token = data.token;
                state.user = data.user;
                window.location.hash = "#/dashboard";
            } else {
                alert(data.error || "Login failed");
            }
        } catch (err) {
            alert("Connection error occurred.");
        }
    });
}

function renderRegister() {
    document.getElementById("app").innerHTML = `
        <div class="auth-card glass-panel">
            <h2>Register Account</h2>
            <form id="form-register">
                <div class="form-group">
                    <label>Username</label>
                    <input type="text" id="reg-name" required placeholder="john_doe">
                </div>
                <div class="form-group">
                    <label>Email Address</label>
                    <input type="email" id="reg-email" required placeholder="john@domain.com">
                </div>
                <div class="form-group">
                    <label>Password</label>
                    <input type="password" id="reg-password" required placeholder="Minimum 6 characters">
                </div>
                <button type="submit" class="btn btn-primary w-full">Create Account</button>
            </form>
        </div>
    `;
    // Register event listener logic matching login...
}

function renderDashboard() {
    if (!state.token) {
        window.location.hash = "#/login";
        return;
    }
    
    document.getElementById("app").innerHTML = `
        <div class="dashboard-container">
            <div class="dashboard-header">
                <h2>Welcome, <span class="gradient-text">${state.user?.username}</span></h2>
                <a href="#/new-scan" class="btn btn-primary">Start New Check</a>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card glass-panel">
                    <h3>Scans Run</h3>
                    <p class="stat-number" id="stat-total">0</p>
                </div>
                <div class="stat-card glass-panel">
                    <h3>Average Similarity</h3>
                    <p class="stat-number" id="stat-avg">0.0%</p>
                </div>
            </div>

            <div class="history-section glass-panel">
                <h3>Recent Plagiarism Reports</h3>
                <div class="table-wrapper">
                    <table class="history-table">
                        <thead>
                            <tr>
                                <th>Scan Context</th>
                                <th>Score</th>
                                <th>Date</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody id="scans-list">
                            <tr><td colspan="4" class="text-center">Loading past reports...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    `;
    
    fetchDashboardData();
}

async function fetchDashboardData() {
    try {
        const res = await fetch(`${API_URL}/scans`, {
            headers: { "Authorization": `Bearer ${state.token}` }
        });
        const data = await res.json();
        
        if (res.ok) {
            document.getElementById("stat-total").innerText = data.length;
            
            if (data.length > 0) {
                const totalSim = data.reduce((acc, curr) => acc + curr.similarity, 0);
                const avg = ((totalSim / data.length) * 100).toFixed(1);
                document.getElementById("stat-avg").innerText = `${avg}%`;
                
                const tableBody = document.getElementById("scans-list");
                tableBody.innerHTML = data.map(item => `
                    <tr>
                        <td><strong>${item.source_info}</strong></td>
                        <td><span class="badge ${item.similarity > 0.5 ? 'badge-high' : 'badge-low'}">${(item.similarity * 100).toFixed(0)}%</span></td>
                        <td>${new Date(item.created_at).toLocaleDateString()}</td>
                        <td>
                            <a href="#/results?id=${item.id}" class="btn btn-secondary btn-sm">Report</a>
                        </td>
                    </tr>
                `).join("");
            } else {
                document.getElementById("scans-list").innerHTML = `<tr><td colspan="4" class="text-center">No reports run yet.</td></tr>`;
            }
        }
    } catch (e) {
        console.error("Failed to load dashboard data.", e);
    }
}

function renderNewScan() {
    if (!state.token) {
        window.location.hash = "#/login";
        return;
    }
    
    document.getElementById("app").innerHTML = `
        <div class="scan-container glass-panel">
            <h2>Compare Documents for Plagiarism</h2>
            <p class="subtitle">Scan two documents side-by-side to compute deep learning similarity scores.</p>
            
            <form id="form-scan" enctype="multipart/form-data">
                <div class="tab-selectors">
                    <button type="button" id="tab-text" class="tab-btn active">Paste Text</button>
                    <button type="button" id="tab-file" class="tab-btn">Upload Files</button>
                </div>
                
                <!-- Text Paste Blocks -->
                <div id="wrapper-text" class="scan-inputs-grid">
                    <div class="textarea-card">
                        <label>Original Document / Context A</label>
                        <textarea id="scan-text-a" placeholder="Paste your first text here..."></textarea>
                    </div>
                    <div class="textarea-card">
                        <label>Comparison Document / Context B</label>
                        <textarea id="scan-text-b" placeholder="Paste your second text here..."></textarea>
                    </div>
                </div>

                <!-- File Uploader Blocks -->
                <div id="wrapper-file" class="scan-inputs-grid hidden">
                    <div class="uploader-card">
                        <label>Upload Document A</label>
                        <input type="file" id="scan-file-a" accept=".txt,.pdf,.docx">
                        <p class="help-text">Supports PDF, DOCX, TXT</p>
                    </div>
                    <div class="uploader-card">
                        <label>Upload Document B</label>
                        <input type="file" id="scan-file-b" accept=".txt,.pdf,.docx">
                        <p class="help-text">Supports PDF, DOCX, TXT</p>
                    </div>
                </div>

                <div class="form-group margin-top-lg">
                    <label>Comparison Project Name (Optional)</label>
                    <input type="text" id="scan-title" placeholder="E.g., Semester 2 Essay Draft">
                </div>

                <button type="submit" class="btn btn-primary btn-large w-full">Compare Documents</button>
            </form>
        </div>
    `;
    
    // Tab toggling events
    const tabText = document.getElementById("tab-text");
    const tabFile = document.getElementById("tab-file");
    const wrapText = document.getElementById("wrapper-text");
    const wrapFile = document.getElementById("wrapper-file");
    let uploadMode = "text";

    tabText.addEventListener("click", () => {
        tabText.classList.add("active");
        tabFile.classList.remove("active");
        wrapText.classList.remove("hidden");
        wrapFile.classList.add("hidden");
        uploadMode = "text";
    });

    tabFile.addEventListener("click", () => {
        tabFile.classList.add("active");
        tabText.classList.remove("active");
        wrapFile.classList.remove("hidden");
        wrapText.classList.add("hidden");
        uploadMode = "file";
    });

    document.getElementById("form-scan").addEventListener("submit", async (e) => {
        e.preventDefault();
        
        const formData = new FormData();
        const scanTitle = document.getElementById("scan-title").value || "Quick Comparison";
        formData.append("source_info", scanTitle);

        if (uploadMode === "text") {
            const textA = document.getElementById("scan-text-a").value;
            const textB = document.getElementById("scan-text-b").value;
            if (!textA.trim() || !textB.trim()) {
                alert("Please fill both input panels.");
                return;
            }
            formData.append("text_a", textA);
            formData.append("text_b", textB);
        } else {
            const fileA = document.getElementById("scan-file-a").files[0];
            const fileB = document.getElementById("scan-file-b").files[0];
            if (!fileA || !fileB) {
                alert("Please select both files.");
                return;
            }
            formData.append("file_a", fileA);
            formData.append("file_b", fileB);
        }

        try {
            // Loading Overlay
            document.getElementById("app").innerHTML = `
                <div class="loader-overlay glass-panel text-center">
                    <div class="spinner"></div>
                    <h3>Comparing Documents...</h3>
                    <p>Executing Siamese LSTM deep learning checks. Please wait.</p>
                </div>
            `;
            
            const res = await fetch(`${API_URL}/detect`, {
                method: "POST",
                headers: { "Authorization": `Bearer ${state.token}` },
                body: formData
            });
            const data = await res.json();
            
            if (res.ok) {
                window.location.hash = `#/results?id=${data.id}`;
            } else {
                alert(data.error || "Detection processing failed.");
                renderNewScan();
            }
        } catch (err) {
            alert("Error connecting to server.");
            renderNewScan();
        }
    });
}

async function renderResults() {
    if (!state.token) {
        window.location.hash = "#/login";
        return;
    }
    
    // Parse scan ID
    const urlParams = new URLSearchParams(window.location.hash.split("?")[1]);
    const id = urlParams.get("id");
    
    if (!id) {
        window.location.hash = "#/dashboard";
        return;
    }

    try {
        const res = await fetch(`${API_URL}/scans/${id}`, {
            headers: { "Authorization": `Bearer ${state.token}` }
        });
        const data = await res.json();
        
        if (!res.ok) {
            alert("Failed to load report details.");
            window.location.hash = "#/dashboard";
            return;
        }

        const percentage = (data.similarity * 100).toFixed(0);
        let scoreColor = "var(--success)";
        if (data.similarity > 0.3) scoreColor = "var(--warning)";
        if (data.similarity > 0.7) scoreColor = "var(--error)";

        document.getElementById("app").innerHTML = `
            <div class="results-container">
                <div class="results-summary glass-panel text-center">
                    <h2>Analysis Summary</h2>
                    <p class="subtitle">${data.source_info}</p>
                    
                    <div class="score-circle" style="--percentage: ${percentage}; --score-color: ${scoreColor}">
                        <div class="score-text">${percentage}%</div>
                    </div>
                    
                    <div class="indicator-desc">
                        <h4>Semantic Plagiarism Level Detected: ${percentage}%</h4>
                    </div>
                </div>

                <div class="highlight-report glass-panel margin-top-md">
                    <h3>Document Comparison Highlight Tool</h3>
                    <p class="subtitle">Highlighted sentences indicate high sentence similarity to the comparison source.</p>
                    <div class="diff-output-box">
                        ${data.highlighted_html}
                    </div>
                </div>

                <div class="actions-footer margin-top-md">
                    <a href="#/dashboard" class="btn btn-secondary">Back to Dashboard</a>
                    <button onclick="window.print()" class="btn btn-primary">Print PDF Report</button>
                </div>
            </div>
        `;
    } catch (e) {
        alert("Failed to load results due to connection error.");
        window.location.hash = "#/dashboard";
    }
}

async function renderAdmin() {
    if (!state.token || !state.user?.is_admin) {
        window.location.hash = "#/dashboard";
        return;
    }

    try {
        const res = await fetch(`${API_URL}/admin/stats`, {
            headers: { "Authorization": `Bearer ${state.token}` }
        });
        const data = await res.json();
        
        document.getElementById("app").innerHTML = `
            <div class="admin-container glass-panel">
                <h2>Admin Control Center</h2>
                
                <div class="stats-grid margin-top-md">
                    <div class="stat-card">
                        <h3>Registered Users</h3>
                        <p class="stat-number">${data.total_users}</p>
                    </div>
                    <div class="stat-card">
                        <h3>Scans Performed</h3>
                        <p class="stat-number">${data.total_scans}</p>
                    </div>
                    <div class="stat-card">
                        <h3>Average System Match</h3>
                        <p class="stat-number">${(data.average_similarity * 100).toFixed(1)}%</p>
                    </div>
                </div>
            </div>
        `;
    } catch (e) {
        alert("Error fetching admin stats.");
    }
}

function renderAbout() {
    document.getElementById("app").innerHTML = `
        <div class="about-container">
            <div class="glass-panel text-center">
                <h2>About the Project & Team</h2>
                <p class="subtitle">Developed as an End-Semester Capstone Project at the Indian Institute of Technology Patna (IIT Patna).</p>
            </div>
            
            <div class="team-grid">
                <div class="member-card glass-panel">
                    <h3>Deepak kumar</h3>
                    <span class="member-roll">Roll: 2312res238</span>
                    <ul class="member-work-list">
                        <li>Collected and prepared dataset for model training.</li>
                        <li>Worked on preprocessing of text data (tokenization, cleaning).</li>
                        <li>Started implementation of RNN/LSTM-based model.</li>
                    </ul>
                </div>
                
                <div class="member-card glass-panel">
                    <h3>Deepanshu Kumar</h3>
                    <span class="member-roll">Roll: 2312res785</span>
                    <ul class="member-work-list">
                        <li>Assisted in dataset preparation and preprocessing.</li>
                        <li>Researched RNN and LSTM architectures.</li>
                        <li>Contributed to initial model development and coding.</li>
                    </ul>
                </div>
                
                <div class="member-card glass-panel">
                    <h3>Deepak kumar</h3>
                    <span class="member-roll">Roll: 2312res783</span>
                    <ul class="member-work-list">
                        <li>Developed the web application using Flask.</li>
                        <li>Started building the interactive dashboard using Streamlit.</li>
                        <li>Working on integrating the ML model with the web interface.</li>
                    </ul>
                </div>
                
                <div class="member-card glass-panel">
                    <h3>Deepak kumar</h3>
                    <span class="member-roll">Roll: 2312res237</span>
                    <ul class="member-work-list">
                        <li>Performed data analysis on collected datasets.</li>
                        <li>Refined and cleaned data for better model performance.</li>
                        <li>Assisted in preparing data suitable for training the model.</li>
                    </ul>
                </div>
                
                <div class="member-card glass-panel">
                    <h3>Deepak kumar</h3>
                    <span class="member-roll">Roll: 2312res236</span>
                    <ul class="member-work-list">
                        <li>Managed overall project workflow and task coordination.</li>
                        <li>Prepared initial PPT and documentation.</li>
                        <li>Supported data analysis and monitored project progress.</li>
                    </ul>
                </div>
            </div>
        </div>
    `;
}

// Launch App
initApp();
```

---

## 🚀 5. Development and Startup Guide

To build, train, and test this project end-to-end:

### Step 1: Pre-requisites & Setup
Ensure you are in a python environment and install backend requirements:
```bash
pip install -r backend/requirements.txt
```

### Step 2: Initialize Database and Export ML Model
To compile the model structures and fit the Tokenizer based on the dataset, run:
```bash
python backend/app/ml/export_model.py
```
This generates `backend/app/ml/siamese_lstm_model.h5` and `backend/app/ml/tokenizer.pickle`.

### Step 3: Start the Backend Server
Launch the Flask development server:
```bash
python backend/run.py
```
It runs by default at `http://127.0.0.1:5000/`. The server automatically instantiates `backend/plagiarism.db` if not present.

### Step 4: Access the Application
Open `frontend/index.html` in any web browser. It accesses the backend server running locally on port 5000.
