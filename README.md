# Final Project for the Machine Learning for Natural Language Processing course at ENSAE Paris

This project has been done by Chloé SEKKAT (ENSAE \& ENS Paris-Saclay) and Jocelyn BEAUMANOIR (ENSAE \& ESSEC).  

# Problem framing

In this project we chose to take a **scientific** approach in which we want the performances of CANINE, the first pre-trained 
tokenization-free and vocabulary-free encoder, that operates directly on character sequences without explicit tokenization.
It seeks to generalize beyond the orthographic forms encountered during pre-training. We want to compare its performances
on several downstream tasks and against several SOTA models. We will focus on two main fields: Question Answering (QA) \& 
Sentiment Analysis/Classification (SC). 

For QA, we are interested in 5 downstream tasks: simple extractive QA on SQuADv2, generalization in zero-shot transfer
settings on multilingual data, robustness to noise, domain adaptation and resistance to adversarial attacks.

For SC, we are primarily interested in binary classification using the well-known SST-2 dataset and robustness to noise 
artificially added to this dataset. We are also interested in more real-life settings dataset (Sentiment140).

Each time the experiment protocol is similar: 

- set seed (for reproducibility)
- encode the dataset using the tokenizer associated to each model (proposed by HuggingFace)
- fed the tokenized data to the model
- training/evaluation loop
- monitor validation loss/accuracy
- use early stopping
- predict/evaluation on test set once the best model has been found
- analyse the predictions and errors of the model
- compare to other models
- build intuition

# Datasets

Here is a list of the datasets we considered, depending on the task.

## Question Answering

- SQuADv2
- XQuAD
- CUAD
- dynabench/qa

## Sentiment Analysis

- SST2 (part of GLUE benchmark)
- Sentiment140 (1.6 billon of tweets)

# Descriptive statistics

Please, take a look at the Colab notebook: 

- [ ] TODO: INSERT LINK

# Embedding techniques \& Task specific modeling

For all considered downstream tasks, we evaluated CANINE against 5 other models, both unilingual and multilingual. Studied
models are:

- CANINE
- BERT
- mBERT
- DistilBERT
- RoBERTa
- XLM-RoBERTa

For each model, we used the associated tokenizer provided by the Hugging Face library. 

Note that CANINE does not require tokenization, instead it operates at the character level using Unicode (``ord(c)``
in Python).

## Question Answering: regression head

To each core model, we added one linear layer (of shape (768,2) for CANINE for instance) to predict start and end logits 
for minimal span answer (given a context and a question, extract the passage with the answer in the context). Note that
the trickiest part was to implement a functional tokenizer for CANINE since the one provided by HG is not suited for QA.
If you are interested, please refer to the code in ``source/question_answering/question_answering/dataset_tokenizers/dataset_character_based_tokenizer.py``.

## Sentiment Analysis: classification head

To each model, we added a linear layer (of shape (768, num_labels) for CANINE e.g.) to predict logits over the number of 
classes in the dataset (2 for SST2 and 3 for Sentiment140), with a dropout layer with $p=0.1$ or $0.2$ depending on the 
model. The implementation was quite straightforward and in the final pipeline we used the architecture provided by
Hugging Face in order to keep only pipeline clean and fast.

# Baselines

As the goal is to evaluate CANINE and compare its performances to BERT-like models, these models are our baseline.

# Evaluation: both quantitative and qualitative

- Question Answering: F1 score and Exact Match
- Sentiment Classification: Accuracy

Link to the Colab notebook which walks through the errors committed by our models on SST2.

- [ ] TODO: ADD COLAB LINK

# Pre-trained models

All pretrained models and custom datasets are available here:

- Question Answering
- Sentiment Classification 

- [ ] TODO: add links

# Experiments \& Results

To know more about the experiments we have done, we strongly advise you to look at each downstream task ``README.md``

- [QA](https://github.com/chloeskt/nlp_ensae/blob/main/source/question_answering/README.md)
- [SA](https://github.com/chloeskt/nlp_ensae/blob/main/source/sentiment_analysis/README.md)

These ``README.md`` contain relevant information to each task. 

# Future directions

- Work on tokenizers
