# Novelista NLP Project - Assignment 1

## Team Information
- **Team Name:** Care
- **Members:** Azwa Khan, L1F23BSCS0101 | Minahil, L1F23BSCS1323

## Dataset Documentation
- **Source:** Custom scraped dataset, extracted data from pdfs
- **Link:** oceanofpdf.com
- **Size:** 10 documents (text files).
- **License:** Public Domain.
- **Eventual Prediction Target:** The future target for this corpus is to serve as the knowledge base for a Retrieval-Augmented Generation (RAG) system. The cleaned text will be semantically chunked to enable open-domain Question Answering and character relationship extraction (specifically, "gossip" and sentiment analysis regarding character dynamics).

## Project Structure
- `data/raw/`: Contains the original, unprocessed `.txt` files (Ignored by Git).
- `data/processed/`: Will contain the cleaned, analysis-ready corpus.
- `src/`: Contains the reproducible Python scripts for loading and cleaning.
- `notebooks/`: Contains the Jupyter notebooks (or Google Colab) for EDA and tokenization analysis.

## Setup & Reproducibility
1. Clone this repository.
2. Create a virtual environment: `python -m venv venv` and activate it.
3. Install dependencies: `pip install -r requirements.txt`
4. Download the 10 raw `.txt` files and place them in `data/raw/`.
5. Run the data loader: `python src/load_data.py`
