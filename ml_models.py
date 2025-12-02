"""
Machine Learning models and utilities for Sentiment Analysis
"""
import re
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import LatentDirichletAllocation
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Initialize components
vader = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Try to load spaCy model, fallback if not available
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None
    print("Warning: spaCy model 'en_core_web_sm' not found. Aspect extraction will be limited.")


class TextPreprocessor:
    """Text preprocessing utilities"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and preprocess text"""
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def lemmatize_text(text: str) -> str:
        """Lemmatize text"""
        words = text.split()
        lemmatized = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return ' '.join(lemmatized)


class SentimentAnalyzer:
    """Sentiment analysis using multiple methods"""
    
    def __init__(self, method: str = 'vader'):
        """
        Initialize sentiment analyzer
        method: 'vader', 'textblob', or 'logistic'
        """
        self.method = method
        self.vectorizer = None
        self.model = None
    
    def analyze(self, text: str) -> Dict[str, any]:
        """
        Analyze sentiment of text
        Returns: {'sentiment': 'positive'/'negative'/'neutral', 'confidence': float}
        """
        if self.method == 'vader':
            return self._analyze_vader(text)
        elif self.method == 'textblob':
            return self._analyze_textblob(text)
        elif self.method == 'logistic':
            return self._analyze_logistic(text)
        else:
            return self._analyze_vader(text)
    
    def _analyze_vader(self, text: str) -> Dict[str, any]:
        """Analyze using VADER"""
        scores = vader.polarity_scores(text)
        
        # Determine sentiment
        if scores['compound'] >= 0.05:
            sentiment = 'positive'
            confidence = scores['compound']
        elif scores['compound'] <= -0.05:
            sentiment = 'negative'
            confidence = abs(scores['compound'])
        else:
            sentiment = 'neutral'
            confidence = 1 - abs(scores['compound'])
        
        # Normalize confidence to 0-1 range
        confidence = min(max(confidence, 0.0), 1.0)
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'raw_scores': scores
        }
    
    def _analyze_textblob(self, text: str) -> Dict[str, any]:
        """Analyze using TextBlob"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        # Determine sentiment
        if polarity > 0.1:
            sentiment = 'positive'
            confidence = polarity
        elif polarity < -0.1:
            sentiment = 'negative'
            confidence = abs(polarity)
        else:
            sentiment = 'neutral'
            confidence = 1 - abs(polarity)
        
        # Normalize confidence
        confidence = min(max(confidence, 0.0), 1.0)
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'raw_scores': {'polarity': polarity}
        }
    
    def _analyze_logistic(self, text: str) -> Dict[str, any]:
        """Analyze using trained Logistic Regression model"""
        if self.model is None or self.vectorizer is None:
            # Fallback to VADER if model not trained
            return self._analyze_vader(text)
        
        # Preprocess text
        cleaned = TextPreprocessor.clean_text(text)
        lemmatized = TextPreprocessor.lemmatize_text(cleaned)
        
        # Vectorize
        text_vector = self.vectorizer.transform([lemmatized])
        
        # Predict
        prediction = self.model.predict(text_vector)[0]
        probabilities = self.model.predict_proba(text_vector)[0]
        
        # Get confidence
        confidence = max(probabilities)
        
        # Map prediction to sentiment
        sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        sentiment = sentiment_map.get(prediction, 'neutral')
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'raw_scores': {'probabilities': probabilities.tolist()}
        }
    
    def train(self, texts: List[str], labels: List[str]):
        """Train logistic regression model"""
        if len(texts) == 0 or len(labels) == 0:
            return
        
        # Preprocess texts
        processed_texts = []
        for text in texts:
            cleaned = TextPreprocessor.clean_text(text)
            lemmatized = TextPreprocessor.lemmatize_text(cleaned)
            processed_texts.append(lemmatized)
        
        # Vectorize
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(processed_texts)
        
        # Map labels to numbers
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        y = [label_map.get(label.lower(), 1) for label in labels]
        
        # Split data
        if len(set(y)) > 1:  # Check if we have multiple classes
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            self.model = LogisticRegression(max_iter=1000, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            return accuracy
        else:
            # If only one class, create a simple model
            self.model = LogisticRegression(max_iter=1000, random_state=42)
            self.model.fit(X, y)
            return 0.5  # Default accuracy if can't evaluate
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if self.model and self.vectorizer:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'vectorizer': self.vectorizer,
                    'method': self.method
                }, f)
    
    def load_model(self, filepath: str):
        """Load trained model"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.vectorizer = data['vectorizer']
                self.method = data.get('method', 'logistic')
        except FileNotFoundError:
            print(f"Model file {filepath} not found")


class AspectExtractor:
    """Extract aspects from text using spaCy"""
    
    @staticmethod
    def extract_aspects(text: str, max_aspects: int = 3) -> Optional[str]:
        """Extract noun phrases as aspects"""
        if nlp is None:
            # Fallback: simple keyword extraction
            words = TextPreprocessor.clean_text(text).split()
            # Filter out stopwords and short words
            aspects = [w for w in words if len(w) > 3 and w not in stop_words]
            return ', '.join(aspects[:max_aspects]) if aspects else None
        
        doc = nlp(text)
        noun_phrases = []
        
        for chunk in doc.noun_chunks:
            # Filter out common stopwords and short phrases
            if len(chunk.text.split()) <= 3 and len(chunk.text) > 3:
                noun_phrases.append(chunk.text.lower())
        
        # Remove duplicates and return
        unique_aspects = list(set(noun_phrases))[:max_aspects]
        return ', '.join(unique_aspects) if unique_aspects else None


class TopicModeler:
    """LDA Topic Modeling"""
    
    def __init__(self, n_topics: int = 5):
        self.n_topics = n_topics
        self.lda_model = None
        self.vectorizer = None
    
    def fit(self, texts: List[str]):
        """Fit LDA model on texts"""
        if len(texts) == 0:
            return
        
        # Preprocess texts
        processed_texts = []
        for text in texts:
            cleaned = TextPreprocessor.clean_text(text)
            lemmatized = TextPreprocessor.lemmatize_text(cleaned)
            processed_texts.append(lemmatized)
        
        # Vectorize
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = self.vectorizer.fit_transform(processed_texts)
        
        # Fit LDA
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            max_iter=10
        )
        self.lda_model.fit(X)
    
    def predict_topic(self, text: str) -> int:
        """Predict topic for a single text"""
        if self.lda_model is None or self.vectorizer is None:
            return 0
        
        cleaned = TextPreprocessor.clean_text(text)
        lemmatized = TextPreprocessor.lemmatize_text(cleaned)
        text_vector = self.vectorizer.transform([lemmatized])
        
        topic_distribution = self.lda_model.transform(text_vector)[0]
        return int(np.argmax(topic_distribution))
    
    def get_topic_words(self, n_words: int = 10) -> Dict[int, List[str]]:
        """Get top words for each topic"""
        if self.lda_model is None or self.vectorizer is None:
            return {}
        
        feature_names = self.vectorizer.get_feature_names_out()
        topics = {}
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words_idx = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics[topic_idx] = top_words
        
        return topics


def process_reviews(reviews: List[str], analyzer: SentimentAnalyzer, 
                   aspect_extractor: AspectExtractor, topic_modeler: Optional[TopicModeler] = None) -> List[Dict]:
    """
    Process a list of reviews and return sentiment analysis results
    """
    results = []
    
    for review in reviews:
        if not review or not isinstance(review, str):
            continue
        
        # Sentiment analysis
        sentiment_result = analyzer.analyze(review)
        
        # Aspect extraction
        aspect = aspect_extractor.extract_aspects(review)
        
        # Topic prediction
        topic_id = None
        if topic_modeler:
            topic_id = topic_modeler.predict_topic(review)
        
        results.append({
            'review_text': review,
            'sentiment': sentiment_result['sentiment'],
            'confidence': sentiment_result['confidence'],
            'aspect': aspect,
            'topic_id': topic_id
        })
    
    return results

