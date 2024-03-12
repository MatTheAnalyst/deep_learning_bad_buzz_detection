import mlflow
import yaml
import pickle
from models_preprocessing import Models_preprocessing
from bert import Bert
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

def mlflow_run(
    config
    ,run_name
    ,x_train
    ,y_train
    ,x_val
    ,y_val
    ,x_test
    ,y_test
):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(config)

        print("Bert instance\n")
        bert_preprocess = Models_preprocessing(config)
        bert_model = Bert(config)

        print("Building tokenizer\n")
        tokenizer = bert_preprocess.build_pretrained_tokenizer()

        print("Preprocessing data for Bert\n")
        x_train_bert_ready, x_val_bert_ready, x_test_bert_ready, y_train_bert, y_val_bert, y_test_bert = bert_preprocess.preprocessing_bert(tokenizer, x_train, y_train, x_val, y_val, x_test, y_test)

        print("Building Bert\n")
        bert_model.build()
        history = bert_model.fit(x_train_bert_ready, y_train_bert, x_val_bert_ready, y_val_bert)
        bert_model.plot_history(history)

        mlflow.log_artifact(config['figure_path'])

        print("Predictions\n")
        predicted_classes, positive_class_proba = bert_model.predict(x_test_bert_ready)

        mlflow.log_metric("roc_auc", roc_auc_score(y_test_bert.flatten(), positive_class_proba.numpy()))
        mlflow.log_metric("accuracy", accuracy_score(y_test_bert.flatten(), predicted_classes))
        mlflow.log_metric("f1", f1_score(y_test_bert.flatten(), predicted_classes))

        mlflow.tensorflow.log_model(bert_model.model, run_name)

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

with open(config['setup']['x_train_sampled_path'], 'rb') as f:
    x_train = pickle.load(f)

with open(config['setup']['x_val_sampled_path'], 'rb') as f:
    x_val = pickle.load(f)

with open(config['setup']['x_test_sampled_path'], 'rb') as f:
    x_test = pickle.load(f)

with open(config['setup']['y_train_preprocessed_path'], 'rb') as f:
    y_train = pickle.load(f)

with open(config['setup']['y_val_preprocessed_path'], 'rb') as f:
    y_val = pickle.load(f)

with open(config['setup']['y_test_preprocessed_path'], 'rb') as f:
    y_test = pickle.load(f)

RUN_NAME = 'bert_no_cleaning'

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Détectez les Bad Buzz grace au Deep Learning")

mlflow_run(
    config = {**config['setup'], **config['preprocessing'], **config['bert']}
    ,run_name=RUN_NAME
    ,x_train=x_train
    ,y_train=y_train
    ,x_val=x_val
    ,y_val=y_val
    ,x_test=x_test
    ,y_test=y_test
)