# BERTopic Implementation for Spanish Literary Journals

Topic modeling pipeline using BERTopic to analyze thematic structures in Spanish-language literary criticism.

## Overview

This repository provides a complete implementation for:
- Training a BERTopic model on a corpus of academic articles
- Analyzing topic evolution over time
- Comparing topic distributions between print and digital publishing eras

## Repository Structure

```
├── code/
│   ├── bertopic_training_comentado.py          # Model training
│   ├── topic_evolution_correlation_comentado.py # Diachronic analysis
│   ├── digital_transition_analysis_comentado.py # Print vs. digital analysis
│   └── data_extraction/                         # PDF preprocessing utilities
│
├── input_data/
│   ├── articles.zip        # Text corpus
│   └── metadata_map.json   # Article metadata
│
└── output/                 # Generated results and visualizations
```

## Requirements

```bash
pip install -r requirements.txt
python -m spacy download es_core_news_lg
```

## Usage

1. Set `directory_path` in `bertopic_training_comentado.py` to your corpus folder
2. Run the training script to generate the model and outputs
3. Run the analysis scripts from a directory containing the model outputs
