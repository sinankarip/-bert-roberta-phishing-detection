# BERT-RoBERTa Phishing Detection

## Dataset

- The dataset used for training and evaluation is included in the repository as `archive.zip`.
- Original source: [Kaggle - Phishing Emails Dataset](https://www.kaggle.com/datasets/subhajournal/phishingemails)

## Project Overview

This project aims to detect phishing attacks in emails. The goal is to distinguish both classic phishing emails and subtle "grey area" social engineering communications.

## Dataset Details

- A mixed dataset consisting of spam, phishing, and legitimate emails.
- Includes emails with typos, corporate impersonations, and ambiguous social engineering attempts.

## Model & Training Process

- **Embedding:** Texts vectorized using BERT.
- **Sentiment Analysis:** Email sentiment extracted using RoBERTa.
- **Model:** Combined features classified using LightGBM.
- **Feature Engineering:** Extensive, including linguistic patterns, signature information, and reply requests.
- **Performance:** F1 score plateaued around 98%.

## How to Use

**Clone Only the `app.ipynb` Notebook via Sparse Checkout**

```bash
mkdir bert-phishing-detection && cd bert-phishing-detection
git init
git remote add origin https://github.com/sinankarip/-bert-roberta-phishing-detection.git
git config core.sparseCheckout true
echo "app.ipynb" >> .git/info/sparse-checkout
git pull origin main
```

**Setup and Run**

1. Open the `app.ipynb` file in Jupyter Notebook.
2. In the first cell, install required packages:
    ```bash
    !pip install -r https://raw.githubusercontent.com/sinankarip/-bert-roberta-phishing-detection/main/requirements.txt
    ```
3. Execute the remaining cells in order to run the application.

## Evaluation

| Email Type         | Accuracy        | Notes                                 |
|--------------------|----------------|---------------------------------------|
| Classic Phishing   | 99% detection  | Link presence and urgency are decisive. |
| Social Engineering | 63% - 88%      | Struggles with subtle social cues.    |
| Grey Area Emails   | Low to Moderate| High uncertainty and misclassifications.|

## Deployment

- The model is deployed with a Gradio UI for a user-friendly interface.
- Users can input text and get immediate phishing probability results.

## Tech Stack

- **Programming Language:** Python 3.10
- **NLP Models:** BERT (embedding), RoBERTa (sentiment analysis) via Huggingface Transformers
- **Machine Learning:** LightGBM
- **Libraries:** scikit-learn, pandas, numpy, transformers, lightgbm
- **Environment:** Ubuntu 22.04, NVIDIA Tesla T4 GPU (training)
- **Deployment:** Gradio UI

## Limitations and Future Work

- Better detection of social engineering signals is needed.
- Integration of sender verification (DKIM/SPF) is required.
- Training data should be enriched with grey area samples.
- Threshold optimization should be performed.

## License

Released under the MIT License.
