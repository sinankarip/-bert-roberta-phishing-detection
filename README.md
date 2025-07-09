# -bert-roberta-phishing-detection

Phishing Detection Model

Extensive feature engineering was performed, incorporating various types of features to improve the model. However, the F1 score plateaued around 98%, so to avoid unnecessary computational overhead, only BERT embeddings and RoBERTa sentiment vectors were retained.
Project Overview

This model is developed to detect email phishing attacks. The goal is to distinguish both classic phishing emails and subtle “grey area” social engineering communications.
Dataset

    A mixed dataset consisting of spam, phishing, and legitimate emails.

    Includes emails with typos, corporate impersonations, and social engineering examples in ambiguous contexts.

Model & Training Process

    Embedding: Texts were vectorized using BERT.

    Sentiment Analysis: Email sentiment was extracted using RoBERTa.

    Model: Features were combined and classified using LightGBM.

    Extensive feature engineering was done, including linguistic patterns, signature information, and reply requests.

    The F1 score plateaued around 98%.

How to Use
Clone Only the app.ipynb Notebook via Sparse Checkout

To avoid cloning the entire repository and fetch only the app.ipynb file, follow these steps:

mkdir bert-phishing-detection && cd bert-phishing-detection
git init
git remote add origin https://github.com/sinankarip/-bert-roberta-phishing-detection.git
git config core.sparseCheckout true
echo "app.ipynb" >> .git/info/sparse-checkout
git pull origin main

Setup and Run

    Open the app.ipynb file in Jupyter Notebook.

    In the first cell, install the required packages by running:

!pip install -r https://raw.githubusercontent.com/sinankarip/-bert-roberta-phishing-detection/main/requirements.txt

    Execute the remaining cells in order to run the application.



Evaluation
Email Type	Accuracy	Notes
Classic Phishing	99% detection	Link presence and urgency are decisive.
Social Engineering	63% - 88%	Struggles with subtle social cues.
Grey Area Emails	Low to Moderate	High uncertainty and misclassifications.
Deployment

The model is deployed with a Gradio UI for a user-friendly interface. Users can input text and get immediate phishing probability results.

Gradio enables fast prototyping and easy demo presentation.
Tech Stack

    Programming Language: Python 3.10

    NLP Models: BERT (embedding), RoBERTa (sentiment analysis) — Huggingface Transformers

    Machine Learning: LightGBM

    Libraries: scikit-learn, pandas, numpy, transformers, lightgbm

    Environment: Ubuntu 22.04, NVIDIA Tesla T4 GPU (training)

    Deployment: Gradio UI

Limitations and Future Work

    Better detection of social engineering signals is needed.

    Integration of sender verification (DKIM/SPF) is required.

    Training data should be enriched with grey area samples.

    Threshold optimization should be performed.

License

Released under the MIT License.
