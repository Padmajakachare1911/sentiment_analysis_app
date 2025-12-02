# Sentiment Analysis Web App with Active Learning

A complete Streamlit-based web application for sentiment analysis with active learning capabilities, admin dashboard, dataset management, and model retraining.

## Features

### 🔐 Authentication
- User login/signup with SQLite-based authentication
- Role-based access control (user/admin)

### 📤 Dataset Management
- Upload CSV files with "review" column
- Load sample dataset for testing
- Project-based organization
- Automatic text preprocessing and cleaning

### 📊 Dashboard
- Sentiment trends over time
- Aspect sentiment distribution
- Confidence score distribution
- Interactive filters (date, sentiment, category)
- Real-time statistics and metrics

### 🎯 Active Learning
- Identify low-confidence predictions (< 50%)
- Manual annotation interface
- Save corrections to training set
- Retrain model with corrected data
- Track model versions and accuracy

### ⚙️ Admin Panel
- Active users monitoring
- Total annotations tracking
- Model accuracy metrics
- Model version history
- Deployment readiness indicators

## Machine Learning Features

- **Sentiment Analysis**: VADER, TextBlob, and Logistic Regression
- **Confidence Scoring**: Probability-based confidence for each prediction
- **Aspect Extraction**: spaCy-based noun phrase extraction
- **Topic Modeling**: LDA (Latent Dirichlet Allocation) for topic discovery
- **Text Preprocessing**: Regex cleaning, stopword removal, lemmatization

## Installation

1. **Clone or download the project**

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download spaCy English model**:
```bash
python -m spacy download en_core_web_sm
```

4. **Run the application**:
```bash
streamlit run app.py
```

## Database Schema

The application uses SQLite with the following tables:

- **users**: User accounts and authentication
- **projects**: Project organization
- **sentiment_results**: Sentiment analysis results
- **manual_annotations**: User corrections for active learning
- **model_versions**: Model training history

## Usage

### 1. Create Account / Login
- Start by creating an account or logging in
- Admin accounts can be created by selecting "admin" role during signup

### 2. Upload Dataset
- Create a new project or select existing one
- Upload a CSV file with a "review" column
- Or use the sample dataset button
- Click "Process Dataset" to analyze reviews

### 3. View Dashboard
- See sentiment trends and distributions
- Filter by date, sentiment, and confidence
- Analyze aspect-based sentiment

### 4. Active Learning
- Review low-confidence predictions
- Correct misclassified sentiments
- Save corrections and retrain model
- Track model improvements

### 5. Admin Panel (Admin only)
- Monitor system metrics
- View model versions
- Check deployment readiness

## Project Structure

```
sentiment_analysis_app/
├── app.py                 # Main Streamlit application
├── database.py            # Database operations
├── ml_models.py          # ML models and utilities
├── active_learning.py    # Active learning pipeline
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── sentiment_app.db     # SQLite database (created automatically)
└── models/              # Saved model files (created automatically)
```

## CSV Format

Your CSV file should have a column named "review":

```csv
review
"This product is amazing!"
"Terrible quality, very disappointed."
"It's okay, nothing special."
```

## Model Retraining

The active learning system:
1. Identifies predictions with confidence < 50%
2. Allows manual correction
3. Combines original data with corrections
4. Retrains Logistic Regression model
5. Saves new model version with accuracy metrics

## Notes

- The application uses VADER by default for sentiment analysis
- Logistic Regression model is trained when retraining is triggered
- spaCy model (`en_core_web_sm`) is required for aspect extraction
- All data is stored locally in SQLite database

## Troubleshooting

**spaCy model not found:**
```bash
python -m spacy download en_core_web_sm
```

**NLTK data missing:**
The application will automatically download required NLTK data on first run.

**Port already in use:**
```bash
streamlit run app.py --server.port 8502
```

## License

This project is provided as-is for educational and development purposes.

