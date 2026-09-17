# Credits

## Dataset

This project uses the **Customer Support on Twitter** dataset published on Kaggle. The assignment requires the dataset to be downloaded locally and prohibits committing the raw dataset to this repository.

## Libraries

The implementation uses open-source Python libraries listed in `requirements.txt`, including pandas, NumPy, scikit-learn, FAISS, sentence-transformers, OpenAI's Python client, python-dotenv, pytest, Ruff, matplotlib, and seaborn.

## LLM Provider

The real LLM integration is designed for Groq's OpenAI-compatible API, using `openai/gpt-oss-120b`. API credentials are supplied locally through environment variables and are never committed.

## Prompts and Borrowed Work

Prompts, evaluation logic, and implementation code are developed specifically for this assignment unless a source is explicitly documented here. Any borrowed code, prompt pattern, dataset-derived artifact, or other external material introduced during development will be added to this file with its source and the relevant usage.

## Attribution Policy

No claim is made that the dataset or third-party libraries are original work of this project author. Their licenses and attribution requirements remain applicable.
