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

## Sentiment Analysis

- SST2 (part of GLUE benchmark)
- Sentiment140 (1.6 million of tweets)
- Multilingual Amazon Reviews Corpus

## Question Answering

- SQuADv2
- XQuAD
- CUAD
- dynabench/qa

# Descriptive statistics

Please, take a look at the Colab notebook: 

<a href="https://drive.google.com/file/d/18Th_ddo6ttR3FGvvgOrBF-a-OJF5VnfX/view?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Data Exploration

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

## Sentiment Analysis: classification head

To each model, we added a linear layer (of shape (768, num_labels) for CANINE e.g.) to predict logits over the number of 
classes in the dataset (2 for SST2 and 3 for Sentiment140), with a dropout layer with $p=0.1$ or $0.2$ depending on the 
model. The implementation was quite straightforward and in the final pipeline we used the architecture provided by
Hugging Face in order to keep only pipeline clean and fast.

## Question Answering: regression head

To each core model, we added one linear layer (of shape (768,2) for CANINE for instance) to predict start and end logits 
for minimal span answer (given a context and a question, extract the passage with the answer in the context). Note that
the trickiest part was to implement a functional tokenizer for CANINE since the one provided by HG is not suited for QA.
If you are interested, please refer to the code in ``source/question_answering/question_answering/dataset_tokenizers/dataset_character_based_tokenizer.py``.

# Baselines

As the goal is to evaluate CANINE and compare its performances to BERT-like models, these models are our baseline.

# Evaluation: both quantitative and qualitative

- Sentiment Classification: Accuracy
- Question Answering: F1 score and Exact Match

Link to the Colab notebook which walks through the errors committed by our models on SST2.

- [ ] TODO: ADD COLAB LINK

# Pre-trained models

Most pretrained models and custom datasets are available here:

- [Sentiment Classification](https://drive.google.com/drive/folders/1HjKQ_C_EoBDncjA3nJ-IlgjwhQe4bKTO?usp=sharing) 
- [Question Answering](https://drive.google.com/drive/folders/1L9Su25qatgdmoz-rZbeY_tA2bXq9T9EG?usp=sharing)

Note that not all pretrained models are available due to limited storage our Google Drive. Contact the owner of this 
repository for questions if needed.

# Experiments \& Results

To know more about the experiments we have done, we **strongly** advise you to look at each downstream task ``README.md``

- [SA](https://github.com/chloeskt/nlp_ensae/blob/main/source/sentiment_analysis/README.md)
- [QA](https://github.com/chloeskt/nlp_ensae/blob/main/source/question_answering/README.md)

These ``README.md`` contain relevant information to each task. 

You can also take a look at the notebooks we have created, they serve as showcase for our work:

- [ ] Add links

# Future directions

- Work on tokenizers
- More hyperparameters search for CANINE models (could not afford to do that during this project due to limited time and
compute resources)

# On note on language families

CANINE is a multilingual model, as such we have tested it on several languages both in zero-shot transfer setting and 
when finetuned on the data directly. To gain insights on the results that can be found in each respective ``README.md``
(for QA ans SC), it is interesting to take a look at the genetic proximity of each language by comparing their family and
proximity to English. This is what is given by the following table: 

| Language 	|     Family    	| Proximity with English 	|
|:--------:	|:-------------:	|:----------------------:	|
|  Russian 	|  East Slavic  	|          60.3          	|
|   Dutch  	| West Germanic 	|          27.2          	|
|  German  	| West Germanic 	|          30.8          	|
|  Turkish 	|     Turkic    	|          92.0          	|
|  French  	|    Romance    	|          48.7          	|
|   Thai   	|      Tai      	|          92.9          	|
|  Chinese 	|  Sino-Tibetan 	|          82.4          	|
| Japanese 	|    Japanese   	|          88.3          	|
| Romanian 	|    Romance    	|          54.9          	|
|   Greek  	|     Greek     	|          69.9          	|
|  Spanish 	|    Romance    	|          57.0          	|
|   Hindi  	|     Indic     	|          65.2          	|
|  Arabic  	|    Semitic    	|          83.6          	|

Note on the scores:

- Between 1 and 30: Highly related languages. Protolanguage (common “ancestor”) between several centuries and approx. 2000 years.
- Between 30 and 50: Related languages. Protolanguage approx. between 2000 and 4000 years.
- Between 50 and 70: Remotely related languages. Protolanguage approx. between 4000 and 6000 years.
- Between 70 and 78: Very remotely related languages. Protolanguage approx. older than 6000 years - but high potential of interference with chance ressemblance.
- Between 78 and 100: No recognizable relationship: the few ressemlances measured are more likely to be due to chance than to common origin!

Source: 
- https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md
- http://www.elinguistics.net/Compare_Languages.aspx
