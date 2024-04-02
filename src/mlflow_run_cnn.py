import mlflow
import yaml
import pickle
from models_preprocessing import Models_preprocessing
from cnn import Cnn
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

        print("CNN instance\n")
        cnn_preprocess = Models_preprocessing(config)
        cnn_model = Cnn(config)

        print("Building tokenizer\n")
        tokenizer = cnn_preprocess.build_keras_tokenizer(x_train)

        print("Preprocessing data for CNN\n")
        x_train_cnn_ready, x_val_cnn_ready, x_test_cnn_ready, y_train_cnn, y_val_cnn, y_test_cnn = cnn_preprocess.preprocessing_cnn(tokenizer, x_train, y_train, x_val, y_val, x_test, y_test)

        if config['pretrained_embedding_matrix_path'] is not None:
            print("Building pretrained embedding matrix\n")
            embedding_matrix, nonzero_elements = cnn_preprocess.build_pretrained_embedding_matrix(tokenizer)
            mlflow.log_metric("Percentage of embedded vocab", nonzero_elements)

            print("Building CNN\n")
            cnn_model.build(embedding_matrix)
            history = cnn_model.fit(x_train_cnn_ready, y_train_cnn, x_val_cnn_ready, y_val_cnn)
            mlflow.log_figure(cnn_model.plot_history(history), f'{run_name}_history_loss_accuracy.png')
        else:
            print("Building CNN without pretrained embedding matrix\n")
            cnn_model.build()
            history = cnn_model.fit(x_train_cnn_ready, y_train_cnn, x_val_cnn_ready, y_val_cnn)
            mlflow.log_figure(cnn_model.plot_history(history), f'{run_name}_history_loss_accuracy.png')

        print("Predictions\n")
        y_pred = cnn_model.predict(x_test_cnn_ready)
        predicted_classes = (y_pred >= 0.5).astype(int)

        mlflow.log_metric("accuracy", accuracy_score(y_test_cnn, predicted_classes))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test_cnn, y_pred))
        mlflow.log_metric("f1", f1_score(y_test_cnn, predicted_classes))

        mlflow.tensorflow.log_model(cnn_model.model, run_name)

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

with open(config['setup']['x_train_lemmatized_path'], 'rb') as f:
    x_train = pickle.load(f)

with open(config['setup']['x_val_lemmatized_path'], 'rb') as f:
    x_val = pickle.load(f)

with open(config['setup']['x_test_lemmatized_path'], 'rb') as f:
    x_test = pickle.load(f)

with open(config['setup']['y_train_preprocessed_path'], 'rb') as f:
    y_train = pickle.load(f)

with open(config['setup']['y_val_preprocessed_path'], 'rb') as f:
    y_val = pickle.load(f)

with open(config['setup']['y_test_preprocessed_path'], 'rb') as f:
    y_test = pickle.load(f)

RUN_NAME = 'cnn_lemmatization_word2vec_embedding'

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Détectez les Bad Buzz grace au Deep Learning")

mlflow_run(
    config = {**config['setup'], **config['preprocessing'], **config['cnn']}
    ,run_name=RUN_NAME
    ,x_train=x_train
    ,y_train=y_train
    ,x_val=x_val
    ,y_val=y_val
    ,x_test=x_test
    ,y_test=y_test
)