from pathlib import Path
import numpy as np
from gensim.models import KeyedVectors

# Tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer

class Models_preprocessing():
    def __init__(self, config) -> None:
        self.config = config

    def build_keras_tokenizer(self, x_train):
        tokenizer = Tokenizer(num_words=self.config['tokenizer_max_words'])
        tokenizer.fit_on_texts(x_train)
        return tokenizer

    def build_pretrained_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        return tokenizer

    def build_pretrained_embedding_matrix(self, tokenizer):
        pretrained_embedding_matrix_path = Path(self.config['pretrained_embedding_matrix_path'])

        if not pretrained_embedding_matrix_path.exists():
            print("Le chemin du fichier d'embedding n'est pas spécifié dans config.")
            return None

        # Tri des mots par fréquence
        sorted_words = sorted(tokenizer.word_counts.items(), key=lambda x: x[1], reverse=True)
        top_words = set(word for word, count in sorted_words[:self.config['tokenizer_max_words']])

        # Initialisation de la matrice d'embedding
        embedding_matrix = np.zeros((len(top_words) + 1, self.config['embedding_dim']))

        # Charger le modèle Word2Vec ou GloVe
        if pretrained_embedding_matrix_path.suffix == '.bin':
            # Chargement du modèle Word2Vec binaire
            model = KeyedVectors.load_word2vec_format(str(pretrained_embedding_matrix_path), binary=True)
            for word in top_words:
                if word in model:
                    idx = tokenizer.word_index.get(word)
                    if idx is not None and idx < len(top_words) + 1:
                        embedding_matrix[idx] = model[word][:self.config['embedding_dim']]
        else:
            # Chargement d'un fichier d'embedding texte (comme GloVe)
            with pretrained_embedding_matrix_path.open('r', encoding='utf-8') as f:
                for line in f:
                    word, *vector = line.split()
                    if word in top_words:
                        idx = tokenizer.word_index.get(word)
                        if idx is not None and idx < len(top_words) + 1:
                            embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:self.config['embedding_dim']]

        # Calcul du taux de remplissage de la matrice d'embedding
        nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1)) / (len(top_words) + 1)
        return embedding_matrix, nonzero_elements

    def preprocessing_cnn(self, tokenizer, x_train, y_train, x_val, y_val, x_test, y_test):
        x_train_cnn_enc = tokenizer.texts_to_sequences(x_train)
        x_val_cnn_enc = tokenizer.texts_to_sequences(x_val)
        x_test_cnn_enc = tokenizer.texts_to_sequences(x_test)

        # Padding
        maxlen = self.config['max_length']
        x_train_ready = pad_sequences(x_train_cnn_enc, padding='post', maxlen=maxlen)
        x_val_ready = pad_sequences(x_val_cnn_enc, padding='post', maxlen=maxlen)
        x_test_ready = pad_sequences(x_test_cnn_enc, padding='post', maxlen=maxlen)

        # Transformation des labels en matrice 2D
        y_train_cnn = np.asarray(y_train).astype('float32').reshape((-1,1))
        y_val_cnn = np.asarray(y_val).astype('float32').reshape((-1,1))
        y_test_cnn = np.asarray(y_test).astype('float32').reshape((-1,1))

        return x_train_ready, x_val_ready, x_test_ready, y_train_cnn, y_val_cnn, y_test_cnn

    def preprocessing_bert(self, tokenizer, x_train, y_train, x_val, y_val, x_test, y_test):
        # Tokenisation et préparation des données pour l'entraînement
        tokenized_train = tokenizer(x_train.tolist(), padding=True, truncation=True, max_length=self.config['max_length'], return_tensors="np")
        x_train_ready = {'input_ids': tokenized_train['input_ids'], 'attention_mask': tokenized_train['attention_mask']}

        # Tokenisation et préparation des données pour la validation
        tokenized_val = tokenizer(x_val.tolist(), padding=True, truncation=True, max_length=self.config['max_length'], return_tensors="np")
        x_val_ready = {'input_ids': tokenized_val['input_ids'], 'attention_mask': tokenized_val['attention_mask']}

        # Tokenisation et préparation des données pour le test
        tokenized_test = tokenizer(x_test.tolist(), padding=True, truncation=True, max_length=self.config['max_length'], return_tensors="np")
        x_test_ready = {'input_ids': tokenized_test['input_ids'], 'attention_mask': tokenized_test['attention_mask']}

        # Transformation des labels en matrice 2D
        y_train_cnn = np.asarray(y_train).astype('float32').reshape((-1,1))
        y_val_cnn = np.asarray(y_val).astype('float32').reshape((-1,1))
        y_test_cnn = np.asarray(y_test).astype('float32').reshape((-1,1))

        return x_train_ready, x_val_ready, x_test_ready, y_train_cnn, y_val_cnn, y_test_cnn