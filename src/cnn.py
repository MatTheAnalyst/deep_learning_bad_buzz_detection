"""
Initialisation du modèle : config.yaml

Entrainement du modèle .fit

Evaluation du modèle

fit_tokenizer

Save modèle

Save tokenizer

Mlflow :
Au début : constantes du run mlflow
CLEANING ?
TOKENIZING ?
EMBEDDING MATRIX ?


- Chargement des données
- Preprocessing

"""


import matplotlib.pyplot as plt
from tensorflow.keras.layers import Embedding
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPool1D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, Dense


class Cnn():
    def __init__(self, config):
        self.model = None
        self.config = config

    def build(self, pretrained_embedding_matrix=None):
        self.model = Sequential()

        if pretrained_embedding_matrix is not None:
            self.model.add(Embedding(input_dim=self.config['tokenizer_max_words'] + 1
                                     ,output_dim=self.config['embedding_dim']
                                     ,weights=[pretrained_embedding_matrix]
                                     ,input_length=self.config['max_length']
                                     ,trainable=False))
        else:
            self.model.add(Embedding(input_dim=self.config['tokenizer_max_words'] + 1
                                     ,output_dim=self.config['embedding_dim']
                                     ,input_length=self.config['max_length']))
            
        self.model.add(Dropout(rate=0.5))
        self.model.add(Conv1D(filters=128 ,kernel_size=7, padding="valid", activation="relu", strides=3))
        self.model.add(Conv1D(filters=128 ,kernel_size=7, padding="valid", activation="relu", strides=3))
        self.model.add(GlobalMaxPool1D())
        self.model.add(Dense(units=128, activation="relu"))
        self.model.add(Dropout(rate=0.5))
        self.model.add(Dense(units=1, activation="sigmoid"))
        self.model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])

        return self
    
    def fit(self, x_train, y_train, x_val, y_val):
        if self.model is not None:
            early_stopping = EarlyStopping(patience=6,
                                           min_delta=0.01,
                                           monitor='val_loss',
                                           mode='min',
                                           verbose=1)
        
            lr_reducer = ReduceLROnPlateau(factor=0.1,
                                           cooldown=5,
                                           patience=5,
                                           min_lr=0.1e-5,
                                           monitor='val_loss',
                                           mode='min',
                                           verbose=1)

            callbacks = [early_stopping, lr_reducer]

            history = self.model.fit(x_train, y_train,
                                     epochs=self.config['epochs'],
                                     verbose=1,
                                     validation_data=(x_val, y_val),
                                     callbacks=callbacks,
                                     batch_size=self.config['batch_size'],
                                     workers=self.config['workers'])
            return history
        else:
            print("Le modèle doit être construit avec .build() avant l'entraînement.")

    def plot_history(self, history):
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

        plt.savefig(self.config['figure_path'])        

    def predict(self, x_test):
        if self.model is not None:
            return self.model.predict(x_test)
        else:
            print("Le modèle doit être construit avec .build() avant l'entraînement.")