"""
Setup script for Sentiment Analysis App
"""
import subprocess
import sys
import os

def install_requirements():
    """Install Python requirements"""
    print("Installing Python packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✓ Packages installed successfully!")

def download_spacy_model():
    """Download spaCy English model"""
    print("\nDownloading spaCy English model...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✓ spaCy model downloaded successfully!")
    except subprocess.CalledProcessError:
        print("⚠ Warning: Could not download spaCy model. You can install it manually with:")
        print("  python -m spacy download en_core_web_sm")

def download_nltk_data():
    """Download NLTK data"""
    print("\nDownloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("✓ NLTK data downloaded successfully!")
    except Exception as e:
        print(f"⚠ Warning: Could not download NLTK data: {e}")

def main():
    """Main setup function"""
    print("=" * 50)
    print("Sentiment Analysis App - Setup")
    print("=" * 50)
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ Error: requirements.txt not found!")
        return
    
    # Install requirements
    install_requirements()
    
    # Download spaCy model
    download_spacy_model()
    
    # Download NLTK data
    download_nltk_data()
    
    print("\n" + "=" * 50)
    print("Setup complete! You can now run the app with:")
    print("  streamlit run app.py")
    print("=" * 50)

if __name__ == "__main__":
    main()

