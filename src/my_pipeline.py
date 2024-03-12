from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import mlflow
import json
from keras.preprocessing.text import tokenizer_from_json
from text_preprocessing import Text_preprocessing
from keras.preprocessing.sequence import pad_sequences

MODEL_TYPE = "bi"
MODEL_PATH = "models/bi_lstm/cnn_model_bi_lstm_cleaning"
TOKENIZER_TYPE = 'Tokenizer'
TOKENIZER_PATH = "models/bi_lstm/tokenizer/tokenizer.json"

PADDING = 100

def load_tokenizer(tokenizer_path):
    if TOKENIZER_TYPE == 'AutoTokenizer':
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    elif TOKENIZER_TYPE == 'Tokenizer':
        try:
            with open(tokenizer_path, 'r') as f:
                tokenizer_json = json.load(f)
                tokenizer = tokenizer_from_json(tokenizer_json)
        except:
            raise TypeError("Le tokenizer doit être au format JSON.")
    else:
        raise FileNotFoundError

    return tokenizer

def preprocessing(text_input):
    preprocess = Text_preprocessing()
    input_cleaned = preprocess.clean_and_preprocess(text_input)
    input_tokenized = preprocess.tokenizing(input_cleaned)
    return input_tokenized


def tokenizing(text_preprocessed, tokenizer, PADDING):
    if MODEL_TYPE == "bert":
        text_tokenized = tokenizer(text_preprocessed, return_tensors="np", padding=True, truncation=True, max_length=PADDING)
        id_tokens = text_tokenized["input_ids"]
        attention_mask = text_tokenized["attention_mask"]
        return id_tokens, attention_mask

    else:
        text_tokenized = tokenizer.texts_to_sequences([text_preprocessed])
        text_ready = pad_sequences(text_tokenized, padding='post', maxlen=PADDING)
        return text_ready

def main(text_input: str):
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    input_preprocessed = preprocessing(text_input)
    

    if MODEL_TYPE == "bert":
        id_tokens, attention_mask = tokenizing(input_preprocessed, tokenizer, PADDING)
        model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        outputs = model.predict({'input_ids': id_tokens, 'attention_mask': attention_mask})
        logits = outputs.logits
        probabilities = np.array(tf.nn.softmax(logits, axis=-1)).tolist()[0]
        predicted_class = tf.argmax(probabilities).numpy()

        return probabilities, predicted_class
    
    else:
        tokens = tokenizing(input_preprocessed, tokenizer, PADDING)
        model = mlflow.tensorflow.load_model(MODEL_PATH)
        outputs = model.predict(tokens)
        probabilities = outputs.flatten().tolist()
        predicted_class = (outputs >= 0.5).astype(int)

        return probabilities, predicted_class
