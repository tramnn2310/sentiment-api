from transformers import pipeline
import pickle
import os


class SentimentModel:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        """Initialize with a pre-trained sentiment model"""
        self.model = pipeline("sentiment-analysis", model=model_name)
        self.version = "v1.0"

    def predict(self, text):
        """Predict sentiment for a single text"""
        result = self.model(text)[0]
        return {
            "text": text,
            "sentiment": result['label'],
            "confidence": round(result['score'], 4)
        }

    def predict_batch(self, texts):
        """Predict sentiment for multiple texts"""
        results = self.model(texts)
        return [
            {
                "text": text,
                "sentiment": result['label'],
                "confidence": round(result['score'], 4)
            }
            for text, result in zip(texts, results)
        ]

    def save(self, path="models/sentiment_model.pkl"):
        """Save model metadata"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        metadata = {
            "version": self.version,
            "model_name": "distilbert-base-uncased-finetuned-sst-2-english"
        }
        with open(path, 'wb') as f:
            pickle.dump(metadata, f)