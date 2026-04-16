# Intel Image Classifier — Flask App

Classifies images into 6 categories: **buildings, forest, glacier, mountain, sea, street**
using two pre-trained models — PyTorch (`.pth`) and Keras/TensorFlow (`.keras`).

---

## Project structure

```
App-project/
├── rosly_mamekem_model.pth       ← original model files (kept here as backup)
├── rosly_mamekem_model.keras
└── flask_app/
    ├── app.py                    ← Flask server (inference + routes)
    ├── requirements.txt
    ├── Procfile                  ← for Render / Railway
    ├── README.md
    ├── models/
    │   ├── rosly_mamekem_model.pth
    │   └── rosly_mamekem_model.keras
    └── templates/
        └── index.html            ← full frontend (HTML + Tailwind + JS)
```

---

## Run locally

### 1. Create a virtual environment

```bash
cd flask_app
python3 -m venv venv
```

### 2. Activate it

```bash
# Linux / Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **PyTorch only (lighter):** if you don't need TensorFlow, this is enough.
> TensorFlow is optional — the app still works with just PyTorch.

### 4. Run

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

### Enable auto-reload during development

```bash
FLASK_DEBUG=true python app.py
```

---

## Deploy on Render (free — recommended for school)

1. Push your project to a **GitHub repo** (include the `models/` folder).

2. Go to [render.com](https://render.com) → **New → Web Service**.

3. Connect your repo and fill in:

   | Field | Value |
   |---|---|
   | Root Directory | `flask_app` |
   | Runtime | `Python 3` |
   | Build Command | `pip install -r requirements.txt` |
   | Start Command | `gunicorn app:app` |

4. Click **Deploy** → you get a public URL like `https://your-app.onrender.com`.

> The free tier sleeps after 15 min of inactivity. First request after sleep takes ~30 s. Fine for a school demo.

---

## Deploy on Railway (alternative)

1. Go to [railway.app](https://railway.app) → **New Project → Deploy from GitHub**.
2. Select your repo, set Root Directory to `flask_app`.
3. Railway reads the `Procfile` automatically. Done.

---

## Notes

- **TensorFlow** is a large package (~500 MB). If you only need PyTorch, remove `tensorflow` from `requirements.txt`.
- The app handles the case where a model is not installed — it returns a clear error message instead of crashing.
- Models are trained on **64×64** images with 6 classes.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `5000` | Port (set automatically by Render/Railway) |
| `FLASK_DEBUG` | `false` | Set to `true` for auto-reload during dev |
# AIMS-Computer-vision-app
