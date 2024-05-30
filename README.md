# Continual Lifelong Learning in Neural Information Retrieval

This repository provides the code for the paper **"Advancing Continual Lifelong Learning in Neural Information Retrieval: Definition, Dataset, Framework, and Empirical Evaluation."** This paper has been submitted to _Information Sciences_ and is currently under peer review. You can access a preprint version of the paper on [arXiv](https://arxiv.org/abs/2308.08378).

## Framework and Code Implementation

### Models

The `src/model` directory contains implementations of various neural information retrieval models, which include:

- Traditional models: DRMM, KNRM, CKNRM, DUET
- Pre-trained models: BERTdot, BERTcat, ColBERT

Some implementations were inspired by the [matchmaker](https://github.com/sebastian-hofstaetter/matchmaker) repository.

### Continual Learning Strategies

The `src/agents` directory features implementations of diverse continual learning strategies:

- Regularization-based strategies: L2, EWC, EWCol, MAS
- Replay-based strategies: Naive Rehearsal, GEM

These were developed with reference to the [Continual-Learning-Benchmark](https://github.com/GT-RIPL/Continual-Learning-Benchmark) repository.

### Experimentation Entry Point

The `Interface.py` file serves as the entry point for running our experiments.

## Data and Task Configuration

We designed six subtasks to mimic real-world continual information retrieval scenarios, each associated with different themes: 'IT', 'furnishing', 'food', 'health', 'tourism', 'finance'. For details on how these subtasks were constructed, please refer to our paper.

Data for each task was sourced from the [MSMARCO passage ranking dataset](https://microsoft.github.io/msmarco/).

To streamline the training process, we pre-converted textual information into model tokens. For embedding-based models, tokens correspond to the dictionary "glove.6B.100d.txt"; for BERT-based models, they align with the standard BERT vocab file.

## Example Data and How to Run Experiments

We have selected a small subset of data from each subtask as examples. Please download the sample data from [Google Drive](https://drive.google.com/file/d/1qXRQbk7pDxSEK-KWkgeI5iOmXAlw1Bc0/view?usp=drive_link)

Once downloaded, unzip the data into the designated directory and update the directory paths in the code to match your local setup.

## Future Updates

We plan to release the entire dataset to the research community in the future.


## How to Cite

If you use this code or wish to cite our work in your research, please use the following BibTeX entry:

```bibtex
@article{hou2023advancing,
  title={Advancing continual lifelong learning in neural information retrieval: definition, dataset, framework, and empirical evaluation},
  author={Hou, Jingrui and Cosma, Georgina and Finke, Axel},
  journal={arXiv preprint arXiv:2308.08378},
  year={2023},
  url={https://arxiv.org/abs/2308.08378}
}
