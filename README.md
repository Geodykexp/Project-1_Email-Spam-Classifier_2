# Email Spam Classifier (Naive Bayes)

A simple end-to-end email spam classifier built with scikit-learn (Multinomial Naive Bayes) and served via FastAPI with a minimal HTML frontend.

This project includes:
- A training script that prepares data, trains a Naive Bayes model, evaluates it, and serializes the model to a pickle file.
- A FastAPI application that loads the trained model and exposes an API for predictions.
- A lightweight HTML page that posts text to the API and displays the classification result.


## Project Structure

- `Naive_Bayes_Project_3.py` — Model training pipeline: loads `spam.csv`, cleans/Vectorizes text (TF-IDF), trains a MultinomialNB model, evaluates it, and writes `naive_bayes_3.pkl`.
- `naive_bayes_3.pkl` — Serialized scikit-learn model (Multinomial Naive Bayes) saved by the training script.
- `main.py` — FastAPI app serving two routes:
  - `GET /` returns the `index.html` frontend
  - `POST /predict` accepts text and responds with a spam/ham prediction
- `index.html` ��� Minimal UI to submit an email text and display the predicted label.
- `spam.csv` — Dataset used for training (SMS Spam Collection formatted with columns `v1` and `v2`).


## Requirements

- Python 3.9+ recommended
- Install dependencies:

  pip install fastapi uvicorn scikit-learn pandas numpy seaborn matplotlib

  Notes:
  - `pandas`, `numpy`, `seaborn`, `matplotlib` are primarily required for training/evaluation.
  - The runtime API mainly requires `fastapi`, `uvicorn`, and `scikit-learn`.


## How the Model Is Built (Naive_Bayes_Project_3.py)

Key steps in `Naive_Bayes_Project_3.py`:
- Load and preprocess data from `spam.csv` (renames `v1`->`label`, `v2`->`text`, removes unnamed columns, basic text cleaning via regex).
- Split into train/test using `train_test_split`.
- Vectorize text using `TfidfVectorizer(stop_words='english')` — this fitted vectorizer is stored as a module-level variable `vectorizer`.
- Train a `MultinomialNB` model on the vectorized training data.
- Evaluate with accuracy, confusion matrix, and classification report.
- Serialize the trained model to `naive_bayes_3.pkl` using `pickle`.

Outputs:
- `naive_bayes_3.pkl` — model used in the API.
- In-memory (module-level) `vectorizer` — reused by `main.py` for consistent text transformation.

Important implementation note:
- `main.py` imports `vectorizer` directly from `Naive_Bayes_Project_3`. Importing this module will execute the training pipeline at import time (as currently written). This is convenient for development, but not ideal for production. See "Production Considerations" below for improvements.


## API Service (main.py)

The FastAPI app:
- Loads the trained classifier from `naive_bayes_3.pkl` at startup.
- Imports and uses the `vectorizer` from `Naive_Bayes_Project_3.py` to transform incoming text consistently with training.
- Exposes endpoints:
  - `GET /` — Serves `index.html` as a minimal frontend.
  - `POST /predict` — Accepts a form field `text` and returns JSON with the model prediction. Example response:

    {"prediction": "spam"}

Validation and errors:
- Returns `400` if no input text is provided.
- Returns `500` if vectorization or prediction fails.


## Frontend (index.html)

- Simple form that posts to `/predict` via `multipart/form-data` using `fetch`.
- Displays result with a SPAM / NOT SPAM badge based on the server response.
- Works locally in the same origin as the FastAPI backend.


## Run Locally

1) Ensure dependencies are installed (see Requirements).

2) Ensure a model file exists:
- Option A: Use the included `naive_bayes_3.pkl`.
- Option B: Re-train and regenerate it by running the training script:

  python Naive_Bayes_Project_3.py

  The script will write `naive_bayes_3.pkl` in the project root.

3) Start the FastAPI server (development):

  uvicorn main:app --reload --host 0.0.0.0 --port 8000

4) Open the UI in a browser:

  http://127.0.0.1:8000/

5) Test the API via curl (form-encoded):

  curl -X POST -F "text=Congratulations! You won a prize" http://127.0.0.1:8000/predict

  Expected JSON:

  {"prediction": "spam"}


## Reproducible Training

- Data source: `spam.csv` (SMS Spam Collection style). Ensure it is present in the project directory.
- To tweak preprocessing or model hyperparameters, edit `Naive_Bayes_Project_3.py` accordingly, then rerun it to regenerate the pickle.
- The TF-IDF vectorizer is defined as a module-level variable `vectorizer` in the training script and must be kept consistent with the trained model.


## Production Considerations

To avoid re-running training logic on app start and to decouple the API from the training module side effects, consider:
- Moving vectorizer persistence into a separate artifact (e.g., pickle the fitted vectorizer alongside the model) and load both in `main.py` without importing the training script.
- Refactoring `Naive_Bayes_Project_3.py` so it exposes functions (e.g., `train()`, `load_data()`) and uses `if __name__ == "__main__":` to run training, preventing side effects on import.
- Adding input sanitation and length checks in the API to guard against extremely long inputs or non-text payloads.


## Troubleshooting

- ImportError for scikit-learn or FastAPI: verify the environment and `pip install` commands above.
- 400 error on `/predict`: ensure you are sending the `text` field via form data (`multipart/form-data` or `application/x-www-form-urlencoded`).
- 500 error on `/predict`: check that `naive_bayes_3.pkl` exists and matches the `vectorizer` used. If necessary, re-run `Naive_Bayes_Project_3.py`.
- Unicode/encoding issues reading `spam.csv`: the training script uses `encoding='latin-1'`. Keep this consistent with your data file.


## License

This project is for educational purposes. Add a license file if you plan to distribute or deploy widely.
