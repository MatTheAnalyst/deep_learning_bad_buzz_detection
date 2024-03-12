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
from transformers import TFAutoModelForSequenceClassification
import matplotlib.pyplot as plt
from keras.optimizers.legacy import Adam
from keras.losses import SparseCategoricalCrossentropy
import tensorflow as tf 


class Bert():
    def __init__(self, config):
        self.model = None
        self.config = config

    def build(self):
        # Chargement et compilation du modèle BERT
        self.model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        self.model.compile(optimizer=Adam(learning_rate=3e-5), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

        return self
    
    def fit(self, x_train, y_train, x_val, y_val):
        if self.model is not None:
            history = self.model.fit(x_train, y_train,
                                     epochs=self.config['epochs'],
                                     validation_data=(x_val, y_val),
                                     batch_size=self.config['batch_size'])
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
            y_pred = self.model.predict(x_test)
            # Prédire les logits
            logits = y_pred.logits

            # Appliquer la fonction softmax pour obtenir des probabilités
            probabilities = tf.nn.softmax(logits, axis=-1)
            predicted_classes = tf.argmax(probabilities, axis=-1).numpy()
            
            # Classe positive (classe 1 ?)
            positive_class_proba = probabilities[:, 1]

            return predicted_classes, positive_class_proba
        else:
            print("Le modèle doit être construit avec .build() avant l'entraînement.")