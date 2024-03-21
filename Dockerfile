# Définit l'image de base
FROM python:3.10-bullseye

# Définit le répertoire de travail dans le conteneur
WORKDIR /app

# Copie les fichiers de dépendances
COPY ./src/requirements.txt /app/requirements.txt

# Installe les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copie le reste du code source de l'application dans le conteneur
COPY ./src /app

# Exécute ton application
CMD ["uvicorn", "my_fastapi_app:app", "--host", "0.0.0.0", "--port", "80"]