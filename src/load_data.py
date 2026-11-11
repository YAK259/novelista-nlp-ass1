import os
import pandas as pd
import glob

def load_novels(data_dir='data/raw'):
    """
    Loads all .txt files from the raw data directory into a pandas DataFrame.
    """
    file_paths = glob.glob(os.path.join(data_dir, '*.txt'))
    
    if not file_paths:
        raise FileNotFoundError(f"No .txt files found in {data_dir}. Please add the raw novels.")
    
    data = []
    for file_path in file_paths:
        novel_name = os.path.basename(file_path).replace('.txt', '')
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        data.append({'novel_title': novel_name, 'raw_text': text})
        
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    # Ensure we are running from the root directory
    df = load_novels()
    print(f"Successfully loaded {len(df)} novels.")
    print("\nNovel Titles:")
    print(df[['novel_title']].to_string(index=False))
    print(f"\nTotal raw characters in corpus: {df['raw_text'].apply(len).sum():,}")