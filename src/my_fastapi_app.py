from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import json
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json

MODEL_PATH = "outputs/bi_lstm_data_cleaned_model"
TOKENIZER_PATH = "outputs/tokenizer.json"
PADDING = 100

app = FastAPI()

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
    # Charger le modèle
    model = load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, 'r') as f:
        tokenizer_json = json.load(f)
        tokenizer = tokenizer_from_json(tokenizer_json)

    input = 'start exams hour im twitter cool'

    text_tokenized = tokenizer.texts_to_sequences([input])
    text_ready = pad_sequences(text_tokenized, padding='post', maxlen=PADDING)

    # Faire une prédiction
    predictions = model.predict(text_ready)
    probabilities = predictions.flatten().tolist()
    predicted_class = (predictions >= 0.5).astype(int)

    return {"probabilities": probabilities,
            "predicted_class": predicted_class}

    #probabilities, predicted_class = pipeline.main(request.text)
    #return PredictionResponse(probabilities=probabilities, predicted_class=predicted_class)

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)

"""
suivi de perf dans le temps : save requests users, stocker text des proba inf à un certain niveau. 
système de gestion d'évènement (ex : kafka).
uvicorn my_fastapi_app:app --reload

curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"text": "exemple de texte à analyser"}'

curl -X 'POST' \
  'http://0.0.0.0:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"text": "exemple de texte à analyser"}'

  Eteindre le service nginx
"""
