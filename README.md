# Digit Detector

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Backend-Flask-000000?logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/Model-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![NumPy](https://img.shields.io/badge/Numeric-NumPy-013243?logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/Data-pandas-150458?logo=pandas&logoColor=white)
![HTML5](https://img.shields.io/badge/UI-HTML5%20Canvas-E34F26?logo=html5&logoColor=white)

Handwritten digit recognition web app built with Flask, scikit-learn, and an interactive canvas UI.

![Screenshot](https://raw.githubusercontent.com/vishwaswami24/Digit-Detector/main/Screenshot%202026-03-14%20121149.png)

## Overview

Digit Detector lets you:

- draw a digit directly in the browser
- get a live prediction with confidence scores
- confirm or correct the prediction
- save user feedback for future model improvement
- retrain the model using reference MNIST data plus collected feedback

The app is designed as a small end-to-end ML project with a usable interface, not just a notebook or training script.

## Features

- Browser-based digit drawing canvas
- Prediction confidence and ranked alternatives
- Feedback collection for correct and incorrect guesses
- Retraining endpoint for continuous improvement
- Feedback stats dashboard
- Lightweight test coverage for core Flask routes

## Tech Stack

- Python
- Flask
- scikit-learn
- NumPy
- pandas
- Pillow
- joblib
- HTML, CSS, JavaScript

## Project Structure

```text
Digit Detector/
|-- app.py
|-- train_model.py
|-- digit_modeling.py
|-- requirements.txt
|-- feedback_data.csv
|-- models/
|   |-- digit_model.pkl
|   |-- scaler.pkl
|   `-- mnist_subset.joblib
|-- static/
|   |-- script.js
|   `-- style.css
|-- templates/
|   `-- index.html
`-- tests/
    `-- test_app.py
```

## Getting Started

### 1. Clone the project

```powershell
git clone <your-repo-url>
cd "Digit Detector"
```

### 2. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Train the initial model

```powershell
python train_model.py
```

This generates the model artifacts inside `models/`.

### 5. Run the app

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## How It Works

1. Draw a digit on the canvas.
2. Click `Predict digit`.
3. Review the predicted digit and confidence scores.
4. Confirm the prediction or choose the correct digit.
5. Save feedback to `feedback_data.csv`.
6. Retrain the model when enough feedback has been collected.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Main web interface |
| `POST` | `/predict` | Predict a digit from canvas image data |
| `POST` | `/feedback` | Save prediction feedback |
| `GET` | `/feedback_stats` | Get feedback totals and agreement stats |
| `POST` | `/retrain` | Retrain the model with MNIST + saved feedback |
| `GET` | `/health` | Health and model readiness status |

## Running Tests

```powershell
python -m unittest discover -s tests
```

## Notes

- If the app says the model is missing, run `python train_model.py` first.
- Feedback images are stored in `feedback_data.csv` as base64 strings.
- Retraining uses a cached MNIST subset when available to speed things up.

## Use Cases

- Mini machine learning demo project
- Flask + ML portfolio project
- Beginner-friendly feedback learning example
- Handwritten digit recognizer with a browser UI
