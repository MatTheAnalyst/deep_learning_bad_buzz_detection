import mlflow
import yaml
import pickle
import json
from text_preprocessing import Text_preprocessing
from training_models import Models
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

def mlflow_run(
    config
    ,x_train
    ,y_train
    ,x_val
    ,y_val
    ,x_test
    ,y_test
):
    with mlflow.start_run(run_name=config['mlflow']['run_name']):
        mlflow.log_params(config)
        
        preprocess = Text_preprocessing()
        
        if config['preprocess']['cleaning']:
            print("cleaning\n")
            x_train = preprocess.clean_and_preprocess(x_train)
            x_val = preprocess.clean_and_preprocess(x_val)
            x_test = preprocess.clean_and_preprocess(x_test)

        if config['preprocess']['lemmatizing']:
            print("lemmatizing\n")
            x_train = preprocess.tokenizing(x_train)
            x_val = preprocess.tokenizing(x_val)
            x_test = preprocess.tokenizing(x_test)
        
        print("Configuration\n")
        models = Models(config)

        print("Building tokenizer\n")
        tokenizer = models.build_tokenizer(x_train)
        if config['model']['bert']:
            tokenizer.save_pretrained(config['mlflow']['tokenizer_dir_path'])
            mlflow.log_artifacts(config['mlflow']['tokenizer_dir_path'], artifact_path="tokenizer")
        else:
            tokenizer_json = tokenizer.to_json()
            with open(config['mlflow']['tokenizer_json_path'], "w") as json_file:
                json.dump(tokenizer_json, json_file)
            mlflow.log_artifact(config['mlflow']['tokenizer_json_path'], artifact_path="tokenizer")

        print("Preprocessing\n")
        x_train_cnn_ready, x_val_cnn_ready, x_test_cnn_ready, y_train_cnn, y_val_cnn, y_test_cnn = models.preprocessing_cnn(tokenizer, x_train, y_train, x_val, y_val, x_test, y_test)


        if config['model']['cnn']:
            if config['cnn']['pretrained_embedding_matrix_path'] is not None:
                print("Building embedding matrix\n")
                embedding_matrix, nonzero_elements = models.build_embedding_matrix(tokenizer)
                mlflow.log_metric("Percentage of embedded vocab",nonzero_elements)
                print("Building CNN\n")
                model = models.build_cnn(embedding_matrix)
                model_trained = models.train_cnn(model, x_train_cnn_ready, y_train_cnn, x_val_cnn_ready, y_val_cnn)

            else:
                print("Building CNN\n")
                model = models.build_cnn()
                model_trained = models.train_cnn(model, x_train_cnn_ready, y_train_cnn, x_val_cnn_ready, y_val_cnn)

        
        elif config['model']['bi_lstm']:
            if config['cnn']['pretrained_embedding_matrix_path'] is not None:
                print("Building embedding matrix\n")
                embedding_matrix, nonzero_elements = models.build_embedding_matrix(tokenizer)
                mlflow.log_metric("Percentage of embedded vocab",nonzero_elements)
                print("Training Bi-LSTM\n")
                model_trained = models.bi_lstm(x_train_cnn_ready, y_train_cnn, x_val_cnn_ready, y_val_cnn, embedding_matrix)
            else:
                model_trained = models.bi_lstm(x_train_cnn_ready, y_train_cnn, x_val_cnn_ready, y_val_cnn)

        elif config['model']['bert']:
            print("Training Bert\n")
            model_trained = models.bert(x_train_cnn_ready, y_train_cnn, x_val_cnn_ready, y_val_cnn)

        mlflow.log_artifact(config['mlflow']['figure_path'])

        print("Predictions\n")
        y_pred = model_trained.predict(x_test_cnn_ready)
        
        if config['model']['bert']:
            # Prédire les logits
            logits = y_pred.logits

            # Appliquer la fonction softmax pour obtenir des probabilités
            probabilities = tf.nn.softmax(logits, axis=-1)
            predicted_classes = tf.argmax(probabilities, axis=-1).numpy()

            # Classe positive (classe 1 ?)
            y_proba = probabilities[:, 1]

            mlflow.log_metric("roc_auc", roc_auc_score(y_test_cnn.flatten(), y_proba.numpy()))
            mlflow.log_metric("accuracy", accuracy_score(y_test_cnn.flatten(), predicted_classes))
            mlflow.log_metric("f1", f1_score(y_test_cnn.flatten(), predicted_classes))

        else:
            predicted_classes = (y_pred >= 0.5).astype(int)
            mlflow.log_metric("accuracy", accuracy_score(y_test_cnn, predicted_classes))
            mlflow.log_metric("roc_auc", roc_auc_score(y_test_cnn, y_pred))
            mlflow.log_metric("f1", f1_score(y_test_cnn, predicted_classes))

        mlflow.tensorflow.log_model(model_trained, config['mlflow']['run_name'])

with open(config['sampling']['x_train_sampled_path'], 'rb') as f:
    x_train = pickle.load(f)

with open(config['sampling']['x_val_sampled_path'], 'rb') as f:
    x_val = pickle.load(f)

with open(config['sampling']['x_test_sampled_path'], 'rb') as f:
    x_test = pickle.load(f)

with open(config['sampling']['y_train_sampled_path'], 'rb') as f:
    y_train = pickle.load(f)

with open(config['sampling']['y_val_sampled_path'], 'rb') as f:
    y_val = pickle.load(f)

with open(config['sampling']['y_test_sampled_path'], 'rb') as f:
    y_test = pickle.load(f)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Détectez les Bad Buzz grace au Deep Learning")

mlflow_run(
    config
    ,x_train=x_train
    ,y_train=y_train
    ,x_val=x_val
    ,y_val=y_val
    ,x_test=x_test
    ,y_test=y_test
)