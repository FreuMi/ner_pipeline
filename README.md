# Evaluation

This repository contains the results of our evaluation in the `results` folder, along with the scripts to reproduce these results. Additionally, the datasets used for evaluation are available in the `dataset` folder. Note that the result files have been manually formatted for easier readability.

## Research Question 1 (R1)

**Research question:**  
How accurate is the foundational LLM-based NER approach for extracting entities from RDF graphs with natural language annotations compared to the classical approach of fine-tuning a domain-specific NER model?

To compare the classical fine-tuned NER model with the LLM-based NER approach:

1. Obtain the fine-tuned model from the [NER model repository](https://github.com/FreuMi/NER_Training). Place the output directory of the training script in this directory.
2. The dataset for validation is available [online](https://www.vcharpenay.link/talks/td-sem-interop.html). Relevant natural language annotations have been extracted for simplicity.
3. Install the [Ollama](https://ollama.com/) framework and download the Gemma2:27b model.

Run the following scripts to perform the evaluations:
- `R1_LLM_NER.py`
- `R1_classical_NER.py`

## Research Question 2 (R2)

**Research question:**  
How accurate is the pipeline using Entity Linking (EL) in combination with the disambiguation and self-verification step with knowledge injection in identifying entities compared to an approach using only the LLM?

To execute R2:

1. Install the [Ollama](https://ollama.com/) framework and download the Gemma2:27b model.

Run the following scripts to perform the evaluations:
- `R2_LLM_only.py`
- `R2_pipeline.py`

