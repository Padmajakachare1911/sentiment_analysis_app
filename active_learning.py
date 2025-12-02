"""
Active Learning module for model retraining
"""
import os
import pickle
from datetime import datetime
from typing import List, Dict, Tuple
from ml_models import SentimentAnalyzer, TextPreprocessor
from database import Database


class ActiveLearning:
    """Active Learning pipeline for model improvement"""
    
    def __init__(self, db: Database, model_dir: str = "models"):
        self.db = db
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
    
    def get_uncertain_samples(self, project_id: int = None, threshold: float = 0.50) -> List[Dict]:
        """Get samples with low confidence for manual annotation"""
        return self.db.get_low_confidence_reviews(project_id, threshold)
    
    def save_correction(self, sentiment_result_id: int, corrected_sentiment: str, user_id: int):
        """Save a manual correction"""
        self.db.save_manual_annotation(sentiment_result_id, corrected_sentiment, user_id)
    
    def retrain_model(self, project_id: int = None) -> Dict:
        """
        Retrain model with original data + corrections
        Returns: {'accuracy': float, 'version': str, 'model_path': str}
        """
        # Get training data
        texts, labels = self.db.get_training_data(project_id)
        
        if len(texts) == 0:
            return {
                'success': False,
                'error': 'No training data available'
            }
        
        # Initialize analyzer
        analyzer = SentimentAnalyzer(method='logistic')
        
        # Train model
        accuracy = analyzer.train(texts, labels)
        
        # Generate version number
        version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_path = os.path.join(self.model_dir, f"sentiment_model_{version}.pkl")
        
        # Save model
        analyzer.save_model(model_path)
        
        # Save model version to database
        self.db.save_model_version(
            version_number=version,
            accuracy=accuracy,
            training_samples=len(texts),
            model_path=model_path
        )
        
        return {
            'success': True,
            'accuracy': accuracy,
            'version': version,
            'model_path': model_path,
            'training_samples': len(texts)
        }
    
    def load_active_model(self) -> SentimentAnalyzer:
        """Load the currently active model"""
        versions = self.db.get_model_versions()
        
        # Find active model
        active_model = None
        for version in versions:
            if version.get('is_active') == 1:
                active_model = version
                break
        
        analyzer = SentimentAnalyzer(method='logistic')
        
        if active_model and active_model.get('model_path'):
            model_path = active_model['model_path']
            if os.path.exists(model_path):
                analyzer.load_model(model_path)
            else:
                # Fallback to VADER if model file not found
                analyzer = SentimentAnalyzer(method='vader')
        else:
            # No trained model, use VADER
            analyzer = SentimentAnalyzer(method='vader')
        
        return analyzer
    
    def batch_correct(self, corrections: List[Dict], user_id: int):
        """Save multiple corrections at once"""
        for correction in corrections:
            self.save_correction(
                correction['sentiment_result_id'],
                correction['corrected_sentiment'],
                user_id
            )

