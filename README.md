# VeriCheck - Plagiarism Detection System

**A College Capstone Project for IIT Patna**

A full-stack web application for detecting semantic plagiarism in text documents using a Siamese LSTM neural network.

---

## Project Overview

- **Project Type**: College Capstone Project
- **Institution**: Indian Institute of Technology Patna (IIT Patna)
- **Development Group**: 5-Member Team

### Team Members

| Name | Roll Number | Contributions |
|------|-------------|---------------|
| Deepak Kumar | 2312res238 | Dataset preparation, preprocessing, LSTM model implementation |
| Deepanshu Kumar | 2312res785 | Dataset preparation, RNN/LSTM research, model development |
| Deepak Kumar | 2312res783 | Flask web application, dashboard, ML integration |
| Deepak Kumar | 2312res237 | Data analysis, data cleaning, performance tuning |
| Deepak Kumar | 2312res236 | Workflow management, documentation, progress tracking |

---

## Features

- **User Authentication**: Register, Login, JWT-based sessions
- **Plagiarism Detection**: Compare two text documents or files
- **File Support**: PDF, DOCX, TXT formats
- **Results Dashboard**: View scan history and statistics
- **Admin Panel**: System-wide statistics for administrators
- **LSTM Model**: Siamese LSTM network with 86% accuracy

---

## Project Structure

```
Capstone 2/
├── data.csv                       # Dataset (source_txt, plagiarism_txt, label)
├── plagiarism-checker-nlp.ipynb   # Jupyter notebook with model training
├── train_snli.txt                 # Raw dataset file
├── requirements.txt               # Python dependencies for notebook
│
├── backend/                       # Flask API Server
│   ├── app/
│   │   ├── __init__.py           # Flask app factory
│   │   ├── models.py             # SQLAlchemy database models
│   │   ├── routes.py             # API endpoints
│   │   └── ml/
│   │       ├── model_helper.py   # ML inference helper
│   │       ├── export_model.py   # Script to export model & tokenizer
│   │       ├── siamese_lstm_model.h5  # [EXPORT] Trained model
│   │       └── tokenizer.pickle # [EXPORT] Fitted tokenizer
│   ├── requirements.txt         # Flask backend dependencies
│   └── run.py                    # Development server runner
│
├── frontend/                      # Vanilla JS SPA
│   ├── index.html                # Main HTML
│   ├── style.css                 # Styling
│   └── app.js                    # SPA router & controllers
│
├── Dockerfile.backend            # Backend container
├── docker-compose.yml            # Full stack deployment
├── nginx.conf                   # Nginx configuration
└── README.md                    # This file
```

---

## Quick Start Guide

### Step 1: Install Backend Dependencies

```bash
cd Capstone\ 2
pip install -r backend/requirements.txt
```

### Step 2: Export ML Model

First, ensure the model architecture matches the notebook. Run the export script:

```bash
python backend/app/ml/export_model.py
```

This will create:
- `backend/app/ml/siamese_lstm_model.h5`
- `backend/app/ml/tokenizer.pickle`

### Step 3: Start Backend Server

```bash
python backend/run.py
```

The API will be available at `http://127.0.0.1:5000/api`

### Step 4: Open Frontend

Open `frontend/index.html` in your browser.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Register new user |
| POST | `/api/auth/login` | Login and get JWT token |
| POST | `/api/detect` | Compare two documents for plagiarism |
| GET | `/api/scans` | Get user's scan history |
| GET | `/api/scans/<id>` | Get specific scan details |
| DELETE | `/api/scans/<id>` | Delete a scan |
| GET | `/api/admin/stats` | Get system-wide statistics (admin only) |
| GET | `/api/health` | Health check endpoint |

---

## Docker Deployment (VPS)

### Build and Run

```bash
# Build containers
docker-compose up --build -d

# Check status
docker-compose ps
```

### Access the Application

- **Frontend**: http://YOUR_VPS_IP
- **Backend API**: http://YOUR_VPS_IP:5000/api

### Stop Services

```bash
docker-compose down
```

---

## Model Architecture

The plagiarism detection uses a **Siamese LSTM** neural network:

```
Input A (50 tokens) → Embedding (128) → LSTM (64) ─┐
                                                     │ Subtract → Dense(64) → Dropout → Dense(1) → Sigmoid
Input B (50 tokens) → Embedding (128) → LSTM (64) ─┘
```

- **Embedding Dimension**: 128
- **LSTM Units**: 64
- **Vocabulary Size**: 20,000 (with OOV token)
- **Max Sequence Length**: 50

### Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 86% |
| Precision | 83% |
| Recall | 90% |
| F1-Score | 88% |

---

## Development Notes

### Database

The system uses SQLite (`plagiarism.db`) which is auto-created on first run.

### File Processing

- **PDF**: Uses PyPDF2 for text extraction
- **DOCX**: Uses python-docx for text extraction
- **TXT**: Direct UTF-8 reading

### Fallback Mode

If the ML model files are not found, the system uses SequenceMatcher (difflib) for similarity calculation as a fallback.

---

## Technologies Used

| Layer | Technology |
|-------|------------|
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Backend | Flask (Python 3.12) |
| Database | SQLite |
| ML Framework | TensorFlow/Keras |
| Authentication | JWT (PyJWT) |
| Containerization | Docker |

---

## License

This is an academic project for IIT Patna. All rights reserved by the development team.