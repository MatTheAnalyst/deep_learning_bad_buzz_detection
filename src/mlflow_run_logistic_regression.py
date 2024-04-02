import mlflow
import yaml
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV



def mlflow_run(
    config
    ,run_name
    ,x_train
    ,y_train
    ,x_test
    ,y_test
):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(config)

        print("TfidfVectorizer\n")
        vectorizer = TfidfVectorizer(max_features=config['max_words'])
        x_train_vectorized = vectorizer.fit_transform(x_train)
        x_test_vectorized = vectorizer.transform(x_test)

        vectorizer_path = f"outputs/{run_name}_tfidf_vectorizer.pkl"
        with open(vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)

        print("GridSearchCV")
        lr_model = GridSearchCV(LogisticRegression(), param_grid=config['param_grid'], cv=config['cross_validation'])
        lr_model.fit(x_train_vectorized, y_train)

        mlflow.log_params(lr_model.best_params_)

        y_true = list(y_test)
        y_pred = lr_model.predict(x_test_vectorized)
        y_pred_proba = lr_model.predict_proba(x_test_vectorized)

        mlflow.log_metric("accuracy", accuracy_score(y_true, y_pred))
        mlflow.log_metric("roc_auc", roc_auc_score(y_true, y_pred_proba[:,1]))
        mlflow.log_metric("f1", f1_score(y_true, y_pred))

        mlflow.sklearn.log_model(lr_model, run_name)


with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

with open(config['setup']['x_train_cleaned_path'], 'rb') as f:
    x_train = pickle.load(f)

with open(config['setup']['x_test_cleaned_path'], 'rb') as f:
    x_test = pickle.load(f)

with open(config['setup']['y_train_preprocessed_path'], 'rb') as f:
    y_train = pickle.load(f)

with open(config['setup']['y_test_preprocessed_path'], 'rb') as f:
    y_test = pickle.load(f)

RUN_NAME = 'logistic_regression_no_cleaning'

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Détectez les Bad Buzz grace au Deep Learning")

mlflow_run(
    config = {**config['setup'], **config['logistic_regression']}
    ,run_name=RUN_NAME
    ,x_train=x_train
    ,y_train=y_train
    ,x_test=x_test
    ,y_test=y_test
)