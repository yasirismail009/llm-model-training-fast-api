from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class DocumentAnalytics:
    """Enhanced document analytics with multiple models for different tasks."""
    
    def __init__(self):
        """Initialize analytics models."""
        try:
            # Initialize summarization model
            self.summarizer = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
            self.summarizer_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
            
            # Initialize classification model
            self.classifier = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base")
            self.classifier_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
            
            # Move models to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.summarizer.to(self.device)
            self.classifier.to(self.device)
            
            logger.info(f"Analytics models initialized successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize analytics models: {str(e)}")
            raise
    
    def generate_summary(self, text: str, max_length: int = 150, min_length: int = 40) -> str:
        """Generate a summary of the given text using BART."""
        try:
            inputs = self.summarizer_tokenizer(
                text, 
                max_length=1024, 
                truncation=True, 
                return_tensors="pt"
            ).to(self.device)
            
            summary_ids = self.summarizer.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
            
            return self.summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {str(e)}")
            return ""
    
    def classify_text(self, text: str) -> Dict[str, Any]:
        """Classify the given text using DeBERTa."""
        try:
            inputs = self.classifier_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.classifier(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # Get top 3 predictions
            top3_prob, top3_indices = torch.topk(probabilities, 3)
            
            return {
                "predictions": [
                    {
                        "label": self.classifier.config.id2label[idx.item()],
                        "probability": prob.item()
                    }
                    for prob, idx in zip(top3_prob[0], top3_indices[0])
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to classify text: {str(e)}")
            return {"predictions": []}
    
    def analyze_document(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive document analysis."""
        try:
            summary = self.generate_summary(text)
            classification = self.classify_text(text)
            
            return {
                "summary": summary,
                "classification": classification,
                "metadata": {
                    "model": "BART-CNN + DeBERTa",
                    "device": str(self.device)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze document: {str(e)}")
            return {
                "error": str(e),
                "summary": "",
                "classification": {"predictions": []}
            } 