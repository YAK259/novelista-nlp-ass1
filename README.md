# Novelista NLP Project - Assignment 1

## Team Information
- **Team Name:** [Your Team Name]
- **Members:** [Name 1, Roll No] | [Name 2, Roll No]

## Dataset Documentation
- **Source:** [e.g., Project Gutenberg / Standard Ebooks / Custom scraped dataset]
- **Link:** [Insert URL where the novels were downloaded]
- **Size:** 10 documents (text files).
- **License:** Public Domain (or specify the exact license).
- **Eventual Prediction Target:** The future target for this corpus is to serve as the knowledge base for a Retrieval-Augmented Generation (RAG) system. The cleaned text will be semantically chunked to enable open-domain Question Answering and character relationship extraction (specifically, "gossip" and sentiment analysis regarding character dynamics).

## Project Structure
- `data/raw/`: Contains the original, unprocessed `.txt` files (Ignored by Git).
- `data/processed/`: Will contain the cleaned, analysis-ready corpus.
- `src/`: Contains the reproducible Python scripts for loading and cleaning.
- `notebooks/`: Contains the Jupyter notebooks for EDA and tokenization analysis.

## Setup & Reproducibility
1. Clone this repository.
2. Create a virtual environment: `python -m venv venv` and activate it.
3. Install dependencies: `pip install -r requirements.txt`
4. Download the 10 raw `.txt` files and place them in `data/raw/`.
5. Run the data loader: `python src/load_data.py`