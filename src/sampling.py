import pickle, yaml
import pandas as pd
from src.text_preprocessing import Text_preprocessing
from sklearn.model_selection import train_test_split

import sys
from pprint import pprint


with open("src/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

preprocess = Text_preprocessing()

raw_data = pd.read_csv('data/raw/training.1600000.processed.noemoticon.csv', encoding='latin-1', sep=',', header=None)
raw_data.columns = ['sentiment', 'id', 'date', 'query', 'user_id', 'text']

x = raw_data['text']
y = raw_data['sentiment'].apply(lambda x: 1 if x == 4 else 0)

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size= 0.2, random_state=42)

# Test datasets
x_test, y_test = preprocess.sampling(config['sampling']['x_test_sample_size'], x_test, y_test)

# Validation datasets
x_val, y_val = preprocess.sampling(config['sampling']['x_val_sample_size'], x_train, y_train)

# Remove values in x_train from x_val
x_train = x_train[~x_train.index.isin(x_val.index)]
y_train = y_train[x_train.index]

# Sampling training datasets
x_train, y_train = preprocess.sampling(config['sampling']['x_train_sample_size'], x_train, y_train)

with open(config['sampling']['x_train_sampled_path'], 'wb') as f:
    pickle.dump(x_train, f)

with open(config['sampling']['x_val_sampled_path'], 'wb') as f:
    pickle.dump(x_val, f)

with open(config['sampling']['x_test_sampled_path'], 'wb') as f:
    pickle.dump(x_test, f)

with open(config['sampling']['y_train_sampled_path'], 'wb') as f:
    pickle.dump(y_train, f)

with open(config['sampling']['y_val_sampled_path'], 'wb') as f:
    pickle.dump(y_val, f)

with open(config['sampling']['y_test_sampled_path'], 'wb') as f:
    pickle.dump(y_test, f)