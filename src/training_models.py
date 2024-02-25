import yaml
from pathlib import Path
import numpy as np
from keras.layers import Embedding
from gensim.models import KeyedVectors

# Tensorflow
from keras import Input, Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPool1D, Dense, LSTM, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import matplotlib.pyplot as plt
from keras.optimizers.legacy import Adam
from keras.losses import SparseCategoricalCrossentropy

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

class Models():
    def __init__(self, config):
        self.max_words = config['cnn']['tokenizer_max_words']
        self.config = config


    def build_tokenizer(self, x_train=None):
        if self.config['model']['bert']:
            print("bert tokenizer")
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        else:
            tokenizer = Tokenizer(num_words=self.max_words)
            tokenizer.fit_on_texts(x_train)
        return tokenizer


    def build_embedding_matrix(self, tokenizer):
        filepath = Path(self.config['cnn']['pretrained_embedding_matrix_path'])

        if not filepath.exists():
            print("Le fichier d'embedding spécifié n'existe pas.")
            return None

        # Tri des mots par fréquence
        sorted_words = sorted(tokenizer.word_counts.items(), key=lambda x: x[1], reverse=True)
        top_words = set(word for word, count in sorted_words[:self.config['cnn']['tokenizer_max_words']])

        # Initialisation de la matrice d'embedding
        embedding_matrix = np.zeros((len(top_words) + 1, self.config['cnn']['embedding_dim']))

        # Charger le modèle Word2Vec ou GloVe
        if filepath.suffix == '.bin':
            # Chargement du modèle Word2Vec binaire
            model = KeyedVectors.load_word2vec_format(str(filepath), binary=True)
            for word in top_words:
                if word in model:
                    idx = tokenizer.word_index.get(word)
                    if idx is not None and idx < len(top_words) + 1:
                        embedding_matrix[idx] = model[word][:self.config['cnn']['embedding_dim']]
        else:
            # Chargement d'un fichier d'embedding texte (comme GloVe)
            with filepath.open('r', encoding='utf-8') as f:
                for line in f:
                    word, *vector = line.split()
                    if word in top_words:
                        idx = tokenizer.word_index.get(word)
                        if idx is not None and idx < len(top_words) + 1:
                            embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:self.config['cnn']['embedding_dim']]

        # Calcul du taux de remplissage de la matrice d'embedding
        nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1)) / (len(top_words) + 1)
        return embedding_matrix, nonzero_elements
    

    def preprocessing_cnn(self, tokenizer, x_train, y_train, x_val, y_val, x_test, y_test):
        if self.config['model']['bert']:
            # Tokenisation et préparation des données pour l'entraînement
            tokenized_train = tokenizer(x_train.tolist(), padding=True, truncation=True, max_length=self.config['cnn']['max_lenght'], return_tensors="np")
            x_train_ready = {'input_ids': tokenized_train['input_ids'], 'attention_mask': tokenized_train['attention_mask']}

            # Tokenisation et préparation des données pour la validation
            tokenized_val = tokenizer(x_val.tolist(), padding=True, truncation=True, max_length=self.config['cnn']['max_lenght'], return_tensors="np")
            x_val_ready = {'input_ids': tokenized_val['input_ids'], 'attention_mask': tokenized_val['attention_mask']}

            # Tokenisation et préparation des données pour le test
            tokenized_test = tokenizer(x_test.tolist(), padding=True, truncation=True, max_length=self.config['cnn']['max_lenght'], return_tensors="np")
            x_test_ready = {'input_ids': tokenized_test['input_ids'], 'attention_mask': tokenized_test['attention_mask']}

        else:
            x_train_cnn_enc = tokenizer.texts_to_sequences(x_train)
            x_val_cnn_enc = tokenizer.texts_to_sequences(x_val)
            x_test_cnn_enc = tokenizer.texts_to_sequences(x_test)

            # Padding
            maxlen = self.config['cnn']['max_lenght']
            x_train_ready = pad_sequences(x_train_cnn_enc, padding='post', maxlen=maxlen)
            x_val_ready = pad_sequences(x_val_cnn_enc, padding='post', maxlen=maxlen)
            x_test_ready = pad_sequences(x_test_cnn_enc, padding='post', maxlen=maxlen)

        # Transformation des labels en matrice 2D
        y_train_cnn = np.asarray(y_train).astype('float32').reshape((-1,1))
        y_val_cnn = np.asarray(y_val).astype('float32').reshape((-1,1))
        y_test_cnn = np.asarray(y_test).astype('float32').reshape((-1,1))

        return x_train_ready, x_val_ready, x_test_ready, y_train_cnn, y_val_cnn, y_test_cnn

    def build_cnn(self, embedding_matrix=None):
        inputs = Input(shape=(self.config['cnn']['max_lenght'],), dtype="int64")
        
        if self.config['cnn']['pretrained_embedding_matrix_path'] is not None:
            x = Embedding(
                input_dim = self.config['cnn']['tokenizer_max_words'] + 1
                ,output_dim = self.config['cnn']['embedding_dim']
                ,weights=[embedding_matrix]
                ,input_length=self.config['cnn']['max_lenght']
                ,trainable=False
                )(inputs)
        else:
            x = Embedding(
                input_dim = self.config['cnn']['tokenizer_max_words'] + 1
                ,output_dim = self.config['cnn']['embedding_dim']
                ,input_length=self.config['cnn']['max_lenght']
                )(inputs)

        x = Dropout(self.config['cnn']['build_params']['dropout_rate'])(x)

        for _ in range(self.config['cnn']['build_params']['num_conv_layers']):
            x = Conv1D(self.config['cnn']['build_params']['conv_units']
                       ,self.config['cnn']['build_params']['kernel_size']
                       ,padding="valid"
                       ,activation="relu"
                       ,strides=self.config['cnn']['build_params']['strides'])(x)
        
        x = GlobalMaxPool1D()(x)
        x = Dense(self.config['cnn']['build_params']['dense_units'], activation="relu")(x)
        x = Dropout(self.config['cnn']['build_params']['dropout_rate'])(x)

        predictions = Dense(1, activation="sigmoid", name="predictions")(x)

        model = Model(inputs, predictions)
        model.compile(loss="binary_crossentropy", optimizer=self.config['cnn']['build_params']['optimizer'], metrics=["accuracy"])

        return model
    
    def bi_lstm(self, x_train, y_train, x_val, y_val, embedding_matrix=None):
        # Input for variable-length sequences of integers
        inputs = Input(shape=(self.config['cnn']['max_lenght'],), dtype="int64")

        if self.config['cnn']['pretrained_embedding_matrix_path'] is not None:
            x = Embedding(
                input_dim = self.config['cnn']['tokenizer_max_words'] + 1
                ,output_dim = self.config['cnn']['embedding_dim']
                ,weights=[embedding_matrix]
                ,input_length=self.config['cnn']['max_lenght']
                ,trainable=False
                )(inputs)
        else:
            # Embed each integer in a 128-dimensional vector
            x = Embedding(self.config['cnn']['tokenizer_max_words'] + 1, self.config['cnn']['embedding_dim'])(inputs)

        # Add 2 bidirectional LSTMs
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = Bidirectional(LSTM(64))(x)
        # Add a classifier
        outputs = Dense(1, activation="sigmoid")(x)
        model = Model(inputs, outputs)

        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        history = model.fit(x_train, y_train, 
                  batch_size=self.config['cnn']['train_params']['batch_size'], 
                  epochs=self.config['cnn']['train_params']['epochs'], 
                  validation_data=(x_val, y_val))
        
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        x = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, acc, 'b', label='Training acc')
        plt.plot(x, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(x, loss, 'b', label='Training loss')
        plt.plot(x, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.savefig(self.config['mlflow']['figure_path'])

        return model
    
    def bert(self, x_train, y_train, x_val, y_val):
        # Chargement et compilation du modèle BERT
        model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        model.compile(optimizer=Adam(learning_rate=3e-5), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

        # Entraînement et évaluation
        history = model.fit(x_train, y_train, 
                            epochs=self.config['cnn']['train_params']['epochs'],
                            batch_size=self.config['cnn']['train_params']['batch_size'], 
                            validation_data=(x_val, y_val))
        
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        x = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, acc, 'b', label='Training acc')
        plt.plot(x, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(x, loss, 'b', label='Training loss')
        plt.plot(x, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.savefig(self.config['mlflow']['figure_path'])
        
        return model

    def train_cnn(self, model, x_train, y_train, x_val, y_val):
        # early stopping pour arreter l'apprentissage quand la métrique val_loss ne progresse plus
        early_stopping = EarlyStopping(patience=self.config['cnn']['train_params']['early_stop_patience'],
                                    min_delta=self.config['cnn']['train_params']['early_stop_min_delta'],
                                    monitor= 'val_loss',
                                    mode='min',
                                    verbose=1)
        
        # diminution automatique du learning rate quand la val loss ne progresse plus
        lr_reducer = ReduceLROnPlateau(factor = self.config['cnn']['train_params']['lr_reducer_factor'],
                                    cooldown = self.config['cnn']['train_params']['lr_reducer_cooldown'],
                                    patience = self.config['cnn']['train_params']['lr_reducer_patience'],
                                    min_lr = self.config['cnn']['train_params']['lr_reducer_min_lr'],
                                    monitor= 'val_loss',
                                    mode= 'min',
                                    verbose = 1)

        callbacks = [early_stopping, lr_reducer]

        history = model.fit(x_train, y_train,
                        epochs=self.config['cnn']['train_params']['epochs'],
                        verbose=False,
                        validation_data=(x_val, y_val),
                        callbacks=callbacks,
                        batch_size=self.config['cnn']['train_params']['batch_size'],
                        workers=self.config['cnn']['train_params']['workers'])
        
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        x = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, acc, 'b', label='Training acc')
        plt.plot(x, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(x, loss, 'b', label='Training loss')
        plt.plot(x, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.savefig(self.config['mlflow']['figure_path'])
        
        return model
    
