# # image de base 
# FROM  python:3.11

# # je créé un dossier app
# WORKDIR /app

# ADD  app/ .

# RUN pip install -r requirements.txt

# EXPOSE 8501

FROM python:3.9-slim

WORKDIR /Utilisation_IA_pour_detecter_les_fraudes_a_la_carte_de_credit_pour_MattonSecure

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/streamlit/streamlit-example.git .

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]


