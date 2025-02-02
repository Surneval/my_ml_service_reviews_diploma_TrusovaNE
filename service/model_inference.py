# service/model_inference.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

class SentimentModel:
    def __init__(self, model_path: str = "./model"):
        """
        Initializes the SentimentModel by loading the tokenizer and model.
        
        Args:
            model_path (str): Path to the directory containing the trained model and tokenizer.
        """
        try:
            logging.info(f"Loading tokenizer from {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            logging.info(f"Loading model from {model_path}...")
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.eval()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            logging.info(f"Model loaded successfully on {self.device}.")
        except Exception as e:
            logging.error(f"Error loading model or tokenizer: {e}")
            raise e

    def predict(self, text: str) -> str:
        """
        Predicts the sentiment of the given text.
        
        Args:
            text (str): The movie review text.
        
        Returns:
            str: "positive" or "negative"
        """
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                prediction = torch.argmax(logits, dim=1).item()
            sentiment = "positive" if prediction == 1 else "negative"
            return sentiment
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise e
