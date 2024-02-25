from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pipeline

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
    probabilities, predicted_class = pipeline.main(request.text)
    return PredictionResponse(probabilities=probabilities, predicted_class=predicted_class)

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

"""