from fastapi import FastAPI
from pydantic import BaseModel
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

MODEL_PATH = "outputs/bi_lstm_data_cleaned_model"
TOKENIZER_PATH = "outputs/bi_lstm_data_cleaned_tokenizer.json"
PADDING = 100

app = FastAPI()

# Charger le modèle et le tokenizer au démarrage de l'application
model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, 'r') as f:
    tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)

class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    probabilities: list
    predicted_class: int

@app.get("/")
async def read_root():
    return {"message": "Bienvenue sur mon API FastAPI"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextRequest):
    input = request.text

    text_tokenized = tokenizer.texts_to_sequences([input])
    text_ready = pad_sequences(text_tokenized, padding='post', maxlen=PADDING)

    # Faire une prédiction
    predictions = model.predict(text_ready)
    probabilities = predictions.flatten().tolist()
    predicted_class = (predictions >= 0.5).astype(int)

    return {"probabilities": probabilities,
            "predicted_class": predicted_class}
