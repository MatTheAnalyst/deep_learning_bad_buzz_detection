from pathlib import Path
import subprocess, sys, re, ssl
import pickle, yaml
import nltk
import pandas as pd
from numpy import asarray
from text_preprocessing import Text_preprocessing
from sklearn.model_selection import train_test_split

def setup_environment():
    # Check Python and PiP version.
    verify_python_pip()

    # Install dependencies.
    print("Installation des dépendances...\n")
    try:
        subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
        print("Dépendances installées avec succès.")
    except subprocess.CalledProcessError as e:
        print(f"Une erreur est survenue lors de l'installation des dépendances: {e}")

def setup_nltk():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
        

    def download_nltk_data():
        directory = "utils/nltk_data"
        Path(directory).mkdir(exist_ok=True)

        packages = ['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger']
        for package in packages:
            nltk.download(package, download_dir=str(Path('utils/nltk_data').absolute()))
    
    download_nltk_data()

def verify_python_pip():
    # Check python version.
    print("Check python version")
    python_version = sys.version_info
    if python_version.major != 3 or python_version.minor < 10: 
        raise Exception("Python 3.10 ou une version supérieure est requise.")

    # Check pip version.
    print("Check pip version")
    pip_version = subprocess.run(["pip", "--version"], capture_output=True, text=True).stdout
    pip_version = re.findall(r'pip\s(.*?)\sfrom', pip_version)[0]
    if int(pip_version[0:2]) < 20:
        raise Exception("Pip 20 ou une version supérieure est requise.")

def sampling_data(config):
    preprocess = Text_preprocessing()

    raw_data = pd.read_csv('data/raw/training.1600000.processed.noemoticon.csv', encoding='latin-1', sep=',', header=None)
    raw_data.columns = ['sentiment', 'id', 'date', 'query', 'user_id', 'text']

    x = raw_data['text']
    y = raw_data['sentiment'].apply(lambda x: 1 if x == 4 else 0)

    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size= 0.2, random_state=42)

    # Test datasets.
    x_test, y_test = preprocess.sampling(config['x_test_sample_size'], x_test, y_test)

    # Validation datasets.
    x_val, y_val = preprocess.sampling(config['x_val_sample_size'], x_train, y_train)

    # Remove values in x_train from x_val.
    x_train = x_train[~x_train.index.isin(x_val.index)]
    y_train = y_train[x_train.index]

    # Sampling training datasets.
    x_train, y_train = preprocess.sampling(config['x_train_sample_size'], x_train, y_train)

    with open(config['x_train_sampled_path'], 'wb') as f:
        pickle.dump(x_train, f)

    with open(config['x_val_sampled_path'], 'wb') as f:
        pickle.dump(x_val, f)

    with open(config['x_test_sampled_path'], 'wb') as f:
        pickle.dump(x_test, f)

    with open(config['y_train_sampled_path'], 'wb') as f:
        pickle.dump(y_train, f)

    with open(config['y_val_sampled_path'], 'wb') as f:
        pickle.dump(y_val, f)

    with open(config['y_test_sampled_path'], 'wb') as f:
        pickle.dump(y_test, f)

def data_preprocessing(config):
    print('Load data sampled')
    with open(config['x_train_sampled_path'], 'rb') as f:
        x_train = pickle.load(f)
    with open(config['x_val_sampled_path'], 'rb') as f:
        x_val = pickle.load(f)
    with open(config['x_test_sampled_path'], 'rb') as f:
        x_test = pickle.load(f)
    with open(config['y_train_sampled_path'], 'rb') as f:
        y_train = pickle.load(f)
    with open(config['y_val_sampled_path'], 'rb') as f:
        y_val = pickle.load(f)
    with open(config['y_test_sampled_path'], 'rb') as f:
        y_test = pickle.load(f)

    print(x_train.head())

    # Labels transformation in 2D matrix.
    y_train_cnn = asarray(y_train).astype('float32').reshape((-1,1))
    y_val_cnn = asarray(y_val).astype('float32').reshape((-1,1))
    y_test_cnn = asarray(y_test).astype('float32').reshape((-1,1))

    print('Store labels')
    with open(config['y_train_preprocessed_path'], 'wb') as f:
        pickle.dump(y_train_cnn, f)
    with open(config['y_val_preprocessed_path'], 'wb') as f:
        pickle.dump(y_val_cnn, f)
    with open(config['y_test_preprocessed_path'], 'wb') as f:
        pickle.dump(y_test_cnn, f)

    preprocess = Text_preprocessing()

    print('Clean data')
    x_train_cleaned = x_train.apply(preprocess.clean_comment)
    x_val_cleaned = x_val.apply(preprocess.clean_comment)
    x_test_cleaned = x_test.apply(preprocess.clean_comment)

    print(x_train_cleaned)
    print('Store cleaned data')
    with open(config['x_train_cleaned_path'], 'wb') as f:
        pickle.dump(x_train_cleaned, f)
    with open(config['x_val_cleaned_path'], 'wb') as f:
        pickle.dump(x_val_cleaned, f)
    with open(config['x_test_cleaned_path'], 'wb') as f:
        pickle.dump(x_test_cleaned, f)

    print('Lemmatize data')
    x_train_lemm = preprocess.lemmatized_comment(x_train_cleaned)
    x_val_lemm = preprocess.lemmatized_comment(x_val_cleaned)
    x_test_lemm = preprocess.lemmatized_comment(x_test_cleaned)

    print('Store lemmatized data')
    with open(config['x_train_lemmatized_path'], 'wb') as f:
        pickle.dump(x_train_lemm, f)
    with open(config['x_val_lemmatized_path'], 'wb') as f:
        pickle.dump(x_val_lemm, f)
    with open(config['x_test_lemmatized_path'], 'wb') as f:
        pickle.dump(x_test_lemm, f)

if __name__ == "__main__":
    try:
        print("Loading config file")
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)

        setup_environment()
        setup_nltk()
        sampling_data(config['setup'])
        data_preprocessing(config['setup'])

    except Exception as e:
        print(e)