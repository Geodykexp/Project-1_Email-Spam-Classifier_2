from pathlib import Path
from typing import Optional, Union
import pickle
import uvicorn
import pickle
import os
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import JSONResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from Naive_Bayes_Project_3 import vectorizer, X_train, X_test, y_train, y_test


app = FastAPI(title="Email Spam Classifier (Naive Bayes)")

# Load the trained model from the pickle file
pickle_in = open("naive_bayes_3.pkl", "rb")
model = pickle.load(pickle_in)


@app.get("/")
async def root(request: Request):
    template_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(template_path, "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content, status_code=200)


@app.post("/predict")
async def predict(text: str = Form(...)):
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="No input provided")
    try:
        input_vector = vectorizer.transform([text])
        pred = model.predict(input_vector)[0]
        return JSONResponse(content={"prediction": str(pred)}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")