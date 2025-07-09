import re
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from transformers import pipeline
import pickle
import warnings
warnings.filterwarnings('ignore') 

class SpamDetectionModel:
    def __init__(self, model_path, label_names=None, spam_threshold=0.4) :
        """
        Spam Detection Model with LightGBM
        Detects spam/phishing (0) vs safe/legitimate (1) messages
        
        Args:
            model_path (str): Path to the saved LightGBM model
            label_names (list): List of label names for classification
        """
        self.model = lgb.Booster(model_file=model_path)
        self.label_names = label_names or ['spam/phishing', 'safe/legitimate']  
        self.spam_threshold = spam_threshold 
        
        self._init_pipelines()
        

        self.label_map = {
            "LABEL_0": "negative",
            "LABEL_1": "neutral", 
            "LABEL_2": "positive"
        }
    
    def _init_pipelines(self):
        """Initialize HuggingFace pipelines"""
        try:
        
            self.sentiment_pipeline = pipeline(
                'sentiment-analysis',
                model='cardiffnlp/twitter-roberta-base-sentiment-latest',
                tokenizer='cardiffnlp/twitter-roberta-base-sentiment-latest',
                device=0,  
                return_all_scores=True
            )
            
      
            self.feat_ext = pipeline(
                'feature-extraction', 
                model='bert-base-uncased', 
                device=0  
            )
            
        except Exception as e:
            print(f"Warning: GPU not available, using CPU. Error: {e}")
            
            self.sentiment_pipeline = pipeline(
                'sentiment-analysis',
                model='cardiffnlp/twitter-roberta-base-sentiment-latest',
                tokenizer='cardiffnlp/twitter-roberta-base-sentiment-latest',
                device=-1,
                return_all_scores=True
            )
            
            self.feat_ext = pipeline(
                'feature-extraction', 
                model='bert-base-uncased', 
                device=-1
            )
    
    def clean_text(self, text):
        """
        Clean and preprocess text
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        

        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        
    
        text = re.sub(r'\s+', ' ', text)
        
      
        text = re.sub(r'\b[\w.-]+?@\w+?\.\w+?\b', ' [EMAIL] ', text)
        
      
        text = re.sub(r'https?://\S+|www\.\S+', ' [URL] ', text)
        

        text = re.sub(r'\d{5,}', ' [NUM] ', text)
        

        text = re.sub(r'([!?.]){2,}', r'\1', text)
        
   
        text = re.sub(r'[^\w\s\[\].,!?@-]', '', text)
   
        text = text.lower()
   
        text = text.strip()
        
        return text
    
    def get_sentiment_scores(self, text):
        """
        Get sentiment scores for text
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Sentiment scores {negative, neutral, positive}
        """
        if not isinstance(text, str) or not text.strip():
            return {'negative': 0.0, 'neutral': 0.0, 'positive': 0.0}
 
        results = self.sentiment_pipeline(text[:512])[0]
 
        scores = {}
        for result in results:
            sentiment_name = self.label_map.get(result['label'], result['label'])
            scores[sentiment_name] = result['score']
        
        return scores
    
    def get_embedding(self, text):
        """
        Get BERT embedding for text
        
        Args:
            text (str): Input text
            
        Returns:
            np.array: 768-dimensional embedding vector
        """
        if not isinstance(text, str) or not text.strip():
            return np.zeros(768)
 
        embedding = self.feat_ext(text[:512])[0]

        return np.mean(embedding, axis=0)
    
    def extract_features(self, text):
        """
        Extract all features from text
        
        Args:
            text (str): Input text
            
        Returns:
            np.array: Feature vector (768 BERT + 3 sentiment = 771 dimensions)
        """

        clean_text = self.clean_text(text)

        embeddings = self.get_embedding(clean_text)

        sentiment_scores = self.get_sentiment_scores(clean_text)

        sentiment_feats = np.array([
            sentiment_scores['negative'],
            sentiment_scores['neutral'],
            sentiment_scores['positive']
        ])

        features = np.hstack([embeddings, sentiment_feats])
        
        return features
    
    def predict(self, text):
        """
        Predict class and confidence for single text
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Prediction results with class, confidence, and all probabilities
        """

        features = self.extract_features(text)

        features = features.reshape(1, -1)
        
   
        probabilities = self.model.predict(features, num_iteration=self.model.best_iteration)
        

        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(1, -1)
        
        probabilities = probabilities[0]  
        

        if len(probabilities) == 1:
            
            pos_prob = float(probabilities[0])
            neg_prob = 1.0 - pos_prob
            probabilities = np.array([neg_prob, pos_prob]) 

        spam_prob = float(probabilities[0])
        if spam_prob > self.spam_threshold:
            predicted_class = self.label_names[0]  # spam/phishing
            predicted_class_idx = 0
        else:
            predicted_class = self.label_names[1]  # safe/legitimate
            predicted_class_idx = 1
        
        confidence = float(probabilities[predicted_class_idx])
        
    
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = self.label_names[predicted_class_idx]
        
        
        confidence = float(probabilities[predicted_class_idx])
        
 
        prob_dict = {
            self.label_names[i]: float(probabilities[i]) 
            for i in range(len(self.label_names))
        }
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': prob_dict,
            'is_spam': predicted_class_idx == 0,  
            'is_safe': predicted_class_idx == 1,  
            'spam_probability': float(probabilities[0]),  
            'safe_probability': float(probabilities[1]),  
            'raw_text': text,
            'cleaned_text': self.clean_text(text)
        }
    
    def predict_batch(self, texts):
        """
        Predict classes and confidences for multiple texts
        
        Args:
            texts (list): List of input texts
            
        Returns:
            list: List of prediction results
        """
        results = []
        
        for text in texts:
            result = self.predict(text)
            results.append(result)
        
        return results
    
    def predict_dataframe(self, df, text_column='text'):
        """
        Predict for DataFrame
        
        Args:
            df (pd.DataFrame): Input DataFrame
            text_column (str): Name of text column
            
        Returns:
            pd.DataFrame: DataFrame with prediction results
        """
        results = []
        
        for text in df[text_column]:
            result = self.predict(text)
            results.append(result)
        

        results_df = pd.DataFrame(results)
        

        for col in df.columns:
            if col != text_column:
                results_df[col] = df[col].values
        
        return results_df

