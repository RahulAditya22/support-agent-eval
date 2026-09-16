# Hiver SDE Intern Assignment — Uber Support Agent Evaluation

This repository builds and evaluates an AI support agent for `Uber_Support` using the Customer Support on Twitter dataset.

## Status

Scaffolded. Data processing, intent discovery, retrieval, response drafting, escalation, evaluation, and final validation are added phase-by-phase.

## Stack

- Python 3.11 for a conventional, explainable runtime.
- pandas / NumPy for data processing and metrics support.
- scikit-learn for TF-IDF classification and retrieval baselines.
- matplotlib for reproducible EDA plots.
- pytest + Ruff for testing and code quality.
- python-dotenv for local environment configuration.
- An LLM client is isolated behind small interfaces so all automated tests can run with mocks; no API call is required by CI.

A lightweight TF-IDF approach is preferred over a heavier vector database for the assignment's fast subsample workflow and interview explainability.

## Reproduction

The exact headline reproduction command will be added after the real dataset sample and evaluation set are available.

## Data

The raw Kaggle dataset is intentionally excluded from Git. A small processed sample will be committed after local Kaggle download and sandbox validation.

## Evaluation integrity

The golden set is human-labelled. Model-generated candidate labels may assist formatting or sampling, but the final intent, reply-quality criteria, and escalation labels are decided by the human evaluator.
