import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from unicodedata import normalize
from load_data import load_novels

# Ensure NLTK data is downloaded
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_novel_text(text):
    """
    Applies the cleaning pipeline in the correct order for PDF-extracted novels.
    """
    # 1. Unicode Normalization (NFC)
    # Justification: Standardizes character encoding, crucial for fixing 
    # broken ligatures or weird apostrophes common in PDF extractions.
    text = normalize('NFC', text)
    
    # 2. Noise Removal (Regex tailored to your specific dataset)
    # Pattern A: Remove the specific PDF watermark found in all 10 files.
    text = re.sub(r'OceanofPDF\.com', '', text, flags=re.IGNORECASE)
    
    # Pattern B: Remove stray standalone page numbers or metadata 
    # (e.g., numbers sitting alone on a line from PDF page breaks).
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Pattern C: Normalize excessive whitespace and broken line breaks 
    # caused by the PDF column extraction.
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 3. Case Folding
    # Justification: We lowercase the text because capitalization is not a 
    # useful signal for our future semantic RAG task, and it reduces vocabulary size.
    text = text.lower()
    
    # 4. Remove non-alphabetic characters (keep spaces)
    # Justification: Strips out remaining PDF artifacts, weird punctuation, 
    # and formatting glitches, leaving only pure narrative text.
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 5. Tokenization
    tokens = nltk.word_tokenize(text)
    
    # 6. Stop-word removal
    # Justification: We remove stop words to reduce dimensionality and focus 
    # the future RAG embeddings on content-bearing words (characters, actions, emotions).
    tokens = [word for word in tokens if word not in stop_words]
    
    # 7. Lemmatization
    # Justification: We chose lemmatization over stemming because stemming 
    # aggressively chops off word ends (e.g., 'running' -> 'run'), whereas 
    # lemmatization uses vocabulary analysis to return the actual dictionary root 
    # (e.g., 'better' -> 'good'), preserving the semantic integrity of the novel's narrative.
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Rejoin tokens into a single cleaned string
    cleaned_text = ' '.join(tokens)
    return cleaned_text

def calculate_vocab_size(text_series):
    """Calculates total unique words (types) across the corpus."""
    all_words = []
    for text in text_series:
        all_words.extend(text.split())
    return len(set(all_words))

if __name__ == "__main__":
    print("Loading raw data...")
    df = load_novels()
    
    print("Calculating pre-cleaning vocabulary size...")
    # Quick token count for raw text
    raw_vocab = calculate_vocab_size(df['raw_text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x).lower().split()))
    
    print("Applying cleaning pipeline...")
    df['cleaned_text'] = df['raw_text'].apply(clean_novel_text)
    
    print("Calculating post-cleaning vocabulary size...")
    clean_vocab = calculate_vocab_size(df['cleaned_text'])
    
    print(f"\n--- VOCABULARY REDUCTION REPORT ---")
    print(f"Vocabulary Size BEFORE cleaning: {raw_vocab}")
    print(f"Vocabulary Size AFTER cleaning: {clean_vocab}")
    print(f"Total Words Removed: {raw_vocab - clean_vocab}")
    print(f"Reduction Percentage: {((raw_vocab - clean_vocab) / raw_vocab) * 100:.2f}%\n")
    
    # Task 5: Save the output
    os.makedirs('data/processed', exist_ok=True)
    output_path = 'data/processed/cleaned_novels.parquet'
    df[['novel_title', 'cleaned_text']].to_parquet(output_path)
    print(f"✅ Cleaned data successfully saved to {output_path}")