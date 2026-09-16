# Hiver SDE Intern Assignment — Uber Support Agent Evaluation

This repository builds and evaluates an AI support agent for `Uber_Support` using the Customer Support on Twitter dataset.

## Project Status

Phase 0 scaffold is complete. The implementation will be built and validated incrementally on `feat/agent-pipeline` before review and merge into `main`.

## Assignment Scope

The system will:

1. Reconstruct customer-support conversations from the real Twitter dataset.
2. Derive and document 6–12 support intents from the Uber Support sample.
3. Classify incoming messages into those intents.
4. Retrieve relevant historical resolved threads to ground replies.
5. Draft a concise support response using the selected evidence.
6. Decide whether to auto-handle or escalate, with an explicit reason.
7. Evaluate the system against a 150–250 example human-labelled golden set.

## Stack

- Python 3.11 for a conventional, explainable runtime.
- pandas / NumPy for data processing and metrics.
- scikit-learn for TF-IDF classification and retrieval baselines.
- FAISS and sentence-transformers for semantic retrieval where justified by the real sample.
- Groq `gpt-oss-120b` through an OpenAI-compatible client for the real LLM path.
- python-dotenv for local environment configuration.
- pytest + Ruff for tests and code quality.
- matplotlib / seaborn for reproducible EDA.

The LLM provider is isolated behind `src/llm_client.py` so unit tests and CI can use mocks without requiring an API key.

## Repository Layout

```text
.
├── data/                 # processed/sample data only; raw dataset is ignored
├── src/                  # agent pipeline and supporting modules
├── eval/                 # evaluation metrics, agreement, and golden set
├── report/               # EDA notes, report, and decision log
├── tests/                # unit and smoke tests
├── .github/workflows/    # CI configuration
├── .env.example          # documented local configuration, no secrets
├── .gitignore            # secrets, raw data, caches, and generated artifacts
├── requirements.txt      # pinned dependency ranges
└── LICENSE
```

## Data Handling

The raw Kaggle dataset must not be committed. The local data-download step is performed by the human evaluator. Only an appropriately sized, processed sample and reproducible analysis artifacts will be added to Git.

The selected brand is `Uber_Support`. Dataset volume checks and any discussion of specific date spikes will be based only on the uploaded sample and recorded evidence.

## Evaluation Integrity

The golden set is human-labelled. The human evaluator owns the final intent label, reference-good-reply criteria, and escalation decision for each example. Automated or LLM-generated suggestions may support logistics, but they do not replace the human labels.

## Reproduction

The final README will contain exact setup, test, evaluation, and reproduction commands once the real dataset sample, golden set, and measured results exist. No placeholder performance numbers are used.

## Security

Never commit `.env`, API keys, Kaggle credentials, raw customer-support data, or other secrets. Use `.env.example` as the configuration template.

## Credits

Borrowed datasets, libraries, prompts, and other external work will be explicitly documented in `CREDITS.md` before submission.
