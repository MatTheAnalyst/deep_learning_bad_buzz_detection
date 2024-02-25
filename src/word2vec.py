import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from pathlib import Path
import mlflow

import ssl
import string
import re
import json
from wordcloud import WordCloud

# Scikit-learn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score

# NLTK
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Tensorflow
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPool1D, Dense

from gensim.models import KeyedVectors
word2vec_embedding = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)

class Text_preprocessing():
    def __init__(self):
        pass

    def sampling(self, sample_size:int, values:pd.Series, labels:pd.Series, labels_for_validation:pd.Series=None, stratify:bool=True):
        if len(values) != len(labels):
            raise ValueError("Les tailles de 'values' et 'labels' doivent être identiques.")
        
        if not isinstance(sample_size, int):
            raise TypeError("Sample_size doit être un entier.")
        
        min_label_count = labels.value_counts().min()
        if min_label_count < sample_size:
            raise ValueError(f"Sample_size ne peut pas dépasser {min_label_count}, le nombre minimal d'occurrences dans 'labels'.")

        if labels_for_validation is not None:
            is_in_sample = labels.index.isin(labels_for_validation.index)
            labels = labels[~is_in_sample]

        if stratify:
            distinct_labels = labels.unique()
            labels_sample = pd.Series()
            for value in distinct_labels:
                labels_sample = pd.concat([labels_sample, labels[labels == value].sample(round(sample_size/len(distinct_labels)), random_state=42)])
    
        else:
            labels_sample = labels.sample(sample_size, random_state=42)
        
        values_sample = values.loc[labels_sample.index]

        return values_sample, labels_sample
    
    

    def clean_comment(self, comment:str):
        if not comment or not isinstance(comment, str):
            print(f"{comment} n'est pas un commentaire valide")
            return ""

        # HTML
        comment = re.sub(r'<.*?>', '', comment)

        # ASCII
        comment = re.sub(r'[^\x00-\x7F]+', '', comment)

        # Numbers
        comment = re.sub(r'[0-9]*', '', comment)

        # Removes any words with 3 or more repeated letters
        comment = re.sub(r"(.)\\1{2,}", '', comment)

        # Removes URL and username
        url_username = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
        comment = re.sub(url_username, ' ', str(comment).lower()).strip()

        # Supprimer la ponctuation
        comment = comment.translate(str.maketrans('', '', string.punctuation))

        # Removes any remaining single letter words
        comment = ' '.join([letter for letter in comment.split() if len(letter)>1])

        return comment
    
        
    def get_wordnet_pos(self, treebank_tag):
        """
        Convert treebank POS tags to WordNet POS tags.
        
        Args:
            treebank_tag: The POS tag from treebank.
        Returns:
            A WordNet POS tag corresponding to the given treebank tag.
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def cleaning(self, data:pd.Series):
        """
        Clean and preprocess a comment string.
        
        Args:
            comment: The comment string to be cleaned.
        Returns:
            A cleaned and preprocessed version of the comment.
        """
        # Convertir en minuscules
        data_cleaned = data.str.lower()
        
        data_cleaned = data_cleaned.apply(self.clean_comment)

        return data_cleaned
    
    def tokenizing(self, data:pd.Series):
        """
        Tokenize the comment string into individual words.
        
        Args:
            comment: The comment string to be tokenized.
        Returns:
            A list of words obtained from the comment.
        """
        return data.apply(word_tokenize)
    
    def delete_stopwords(self, tokens:list):
        stop_words = set(stopwords.words('english'))
        return [token for token in tokens if token not in stop_words]
    
    def lemmatized_tokens(self, tokens:list):
        """
        Lemmatize tokens based on their part-of-speech tags.
        
        Args:
            tokens: A list of word tokens.
        Returns:
            A list of lemmatized word tokens.
        """
        pos_tags = pos_tag(tokens)
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(word, pos=self.get_wordnet_pos(pos)) for word, pos in pos_tags]

    def lemmatizing(self, data:pd.Series, delete_stopwords:bool = True):
        """
        Lemmatize tokens based on their part-of-speech tags.
        
        Args:
            tokens: A list of word tokens.
        Returns:
            A list of lemmatized word tokens.
        """
        if delete_stopwords:
            data_lemmatized = data.apply(self.delete_stopwords)
        data_lemmatized = data
        
        return data_lemmatized.apply(self.lemmatized_tokens)
    
def preprocessing_cnn(preprocessing_params, x_train, y_train, x_val, y_val, x_test, y_test):
    preprocess = Text_preprocessing()
    if preprocessing_params['cleaning']:
        x_train = preprocess.cleaning(x_train)
        x_val = preprocess.cleaning(x_val)
        x_test = preprocess.cleaning(x_test)
    
    if preprocessing_params['lemmatizing']:
        x_train = preprocess.tokenizing(x_train)
        x_train = preprocess.lemmatizing(x_train)
        x_val = preprocess.tokenizing(x_val)
        x_val = preprocess.lemmatizing(x_val)
        x_test = preprocess.tokenizing(x_test)
        x_test = preprocess.lemmatizing(x_test)

    number_of_kept_words = preprocessing_params["max_words"]
    tokenizer = Tokenizer(num_words=number_of_kept_words)
    tokenizer.fit_on_texts(x_train)

    dictionary = tokenizer.word_index
    vocab_size = number_of_kept_words + 1

    Path("./outputs").mkdir(exist_ok=True)

    with open("./outputs/dictionary.json", 'w') as f:
        json.dump(dictionary, f)

    x_train_cnn_enc = tokenizer.texts_to_sequences(x_train)
    x_val_cnn_enc = tokenizer.texts_to_sequences(x_val)
    x_test_cnn_enc = tokenizer.texts_to_sequences(x_test)

    # Padding
    maxlen = preprocessing_params["padding_max_len"]
    x_train_cnn_ready = pad_sequences(x_train_cnn_enc, padding='post', maxlen=maxlen)
    x_val_cnn_ready = pad_sequences(x_val_cnn_enc, padding='post', maxlen=maxlen)
    x_test_cnn_ready = pad_sequences(x_test_cnn_enc, padding='post', maxlen=maxlen)

    # Transformation des labels en matrice 2D
    y_train_cnn = np.asarray(y_train).astype('float32').reshape((-1,1))
    y_val_cnn = np.asarray(y_val).astype('float32').reshape((-1,1))
    y_test_cnn = np.asarray(y_test).astype('float32').reshape((-1,1))

    return x_train_cnn_ready, x_val_cnn_ready, x_test_cnn_ready, y_train_cnn, y_val_cnn, y_test_cnn, tokenizer


def build_cnn(params, embedding_matrix):
    inputs = Input(shape=(params['maxlen'],), dtype="int64")

    x = Embedding(params['vocab_size']
        ,params['embedding_dim']
        ,weights=[embedding_matrix]
        ,input_length=params['maxlen']
        ,trainable=False)(inputs)
    
    x = Dropout(params['dropout_rate'])(x)

    for i in range(params['num_conv_layers']):
        x = Conv1D(params['conv_units'], params['kernel_size'], padding="valid", activation="relu", strides=params['strides'])(x)
    
    x = GlobalMaxPool1D()(x)
    x = Dense(params['dense_units'], activation="relu")(x)
    x = Dropout(params['dropout_rate'])(x)

    predictions = Dense(1, activation="sigmoid", name="predictions")(x)

    model = Model(inputs, predictions)
    model.compile(loss="binary_crossentropy", optimizer=params['optimizer'], metrics=["accuracy"])

    return model

def train_cnn(cnn, train_params, x_train, y_train, x_val, y_val):
    # early stopping pour arreter l'apprentissage quand la métrique val_loss ne progresse plus
    early_stopping = EarlyStopping(patience=train_params['early_stop_patience'],
                                min_delta=train_params['early_stop_min_delta'],
                                monitor= 'val_loss',
                                mode='min',
                                verbose=1)

    # sauvegarde automatique du meilleur modèle
    # mode_autosave = ModelCheckpoint('./outputs/checkpoint',
    #                                 save_weights_only=True,
    #                                 save_best_only=True,
    #                                 monitor='val_accuracy',
    #                                 mode="max",
    #                                 verbose=1
    #                                 )

    # diminution automatique du learning rate quand la val loss ne progresse plus
    lr_reducer = ReduceLROnPlateau(factor = train_params['lr_reducer_factor'],
                                cooldown = train_params['lr_reducer_cooldown'],
                                patience = train_params['lr_reducer_patience'],
                                min_lr = train_params['lr_reducer_min_lr'],
                                monitor= 'val_loss',
                                mode= 'min',
                                verbose = 1)

    callbacks = [early_stopping, lr_reducer]

    history = cnn.fit(x_train, y_train,
                    epochs=train_params['epochs'],
                    verbose=False,
                    validation_data=(x_val, y_val),
                    callbacks=callbacks,
                    batch_size=train_params['batch_size'],
                    workers=train_params['workers'])
    
    return cnn

def cnn_mlflow_run(
        name
        ,preprocessing_params
        ,cnn_params
        ,train_params
        ,x_train
        ,y_train
        ,x_val
        ,y_val
        ,x_test
        ,y_test
        ,glove_params=None
):
    with mlflow.start_run(run_name=name):
        mlflow.log_params(preprocessing_params)
        
        mlflow.log_params(cnn_params)
        mlflow.log_params(train_params)
        mlflow.set_tag("model_name", "CNN")

        x_train_cnn_ready, x_val_cnn_ready, x_test_cnn_ready, y_train_cnn, y_val_cnn, y_test_cnn, tokenizer = preprocessing_cnn(preprocessing_params, x_train, y_train, x_val, y_val, x_test, y_test)
        
        sorted_words = sorted(tokenizer.word_counts.items(), key=lambda x: x[1], reverse=True)
        top_words = set([word for word, count in sorted_words[:preprocessing_params["max_words"]]])

        embedding_matrix = np.zeros((cnn_params["vocab_size"], cnn_params["embedding_dim"]))

        found_words = 0

        for word, i in tokenizer.word_index.items():
            if word in top_words and word in word2vec_embedding:
                embedding_matrix[i] = word2vec_embedding[word]
                found_words += 1


        cnn = build_cnn(cnn_params, embedding_matrix)
        mlflow.log_metric("Percentage of covered vocab by embedding",found_words / len(top_words))

        cnn = train_cnn(cnn, train_params, x_train_cnn_ready, y_train_cnn, x_val_cnn_ready, y_val_cnn)
        y_pred = cnn.predict(x_test_cnn_ready)
        predicted_classes = (y_pred >= 0.5).astype(int)

        mlflow.log_metric("accuracy", accuracy_score(y_test_cnn, predicted_classes))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test_cnn, y_pred))
        mlflow.log_metric("f1", f1_score(y_test_cnn, predicted_classes, average='micro'))

        mlflow.tensorflow.log_model(cnn, "cnn_model_without_preprocessing")


df = pd.read_csv('data/raw/training.1600000.processed.noemoticon.csv', encoding='latin-1', sep=',', header=None)

df.columns = ['sentiment', 'id', 'date', 'query', 'user_id', 'text']

x = df['text']
y = df['sentiment'].apply(lambda x: 1 if x == 4 else 0)
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size= 0.2, random_state=42)

preprocess = Text_preprocessing()

# Test datasets
x_test, y_test = preprocess.sampling(60000, x_test, y_test)

# Validation datasets
x_val, y_val = preprocess.sampling(60000, x_train, y_train)

# Remove values in x_train from x_val
x_train = x_train[~x_train.index.isin(x_val.index)]
y_train = y_train[x_train.index]

# Sampling training datasets
x_train, y_train = preprocess.sampling(300000, x_train, y_train)

preprocessing_params = {
    "max_words" : 2000
    ,"padding_max_len" : 100
    ,"cleaning": True
    ,"lemmatizing" : True
}

cnn_params = {
    'maxlen': 100,
    'vocab_size': 2001,
    'embedding_dim': 300,
    'dropout_rate': 0.5,
    'num_conv_layers': 2,
    'conv_units': 128,
    'kernel_size': 7,
    'strides': 3,
    'dense_units': 128,
    'optimizer': 'adam'
}

train_params = {
    'early_stop_patience': 6,
    'early_stop_min_delta': 0.01,
    'lr_reducer_factor': 0.1,
    'lr_reducer_cooldown': 5,
    'lr_reducer_patience': 5,
    'lr_reducer_min_lr': 0.1e-5,
    'epochs': 20,
    'batch_size': 128,
    'workers': 3
}

mlflow.set_tracking_uri("sqlite:///src/mlflow.db")
mlflow.set_experiment("Détectez les Bad Buzz grace au Deep Learning")

cnn_mlflow_run(
        name="CNN_word2vec_embedding"
        ,preprocessing_params=preprocessing_params
        ,cnn_params=cnn_params
        ,train_params=train_params
        ,x_train=x_train
        ,y_train=y_train
        ,x_val=x_val
        ,y_val=y_val
        ,x_test=x_test
        ,y_test=y_test
)