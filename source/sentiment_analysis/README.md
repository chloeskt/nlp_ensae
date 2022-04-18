# Sentiment Classification with CANINE

## Description 

In this section, we are interested in the capacities of CANINE versus BERT-like models such as BERT, mBERT and XLM-RoBERTa 
on Sentiment Classification tasks. CANINE is a pre-trained tokenization-free and vocabulary-free encoder, that operates directly 
on character sequences without explicit tokenization. It seeks to generalize beyond the orthographic forms encountered 
during pre-training.

We evaluate its capacities on sentence classification with binary labels (positive/negative) on SST2 dataset. We have 
whosen this dataset because it is part of the GLUE benchmark and as such is a standard way of evaluating models for 
sentiment classification tasks. We monitor the accuracy obtained by our CANINE model and compare it to BERT, DistilBERT,
mBERT, RoBERTa and XLM-RoBERTa. Note that only mBERT, XLM-RoBERTa and CANINE are pretrained on multilingual data. mBERT and 
CANINE are pretrained on the same data while XLM-RoBERTa was pretrained on 2.5TB of filtered CommonCrawl data containing 
100 languages.

A second experiment is to test the abilities of CANINE to handle noisy inputs such as keyboard errors, misspellings, 
grammar error etc, which are very likely to happen in real life settings. 

Our third experiment consists in confronting CANINE to a more complex and noisy dataset: Sentiment140. It is made of 1.6
million of tweets hence the language used is more informal, prone to abbreviations and colloquialisms. From reading the 
CANINE paper, CANINE is expected to do better than regular token-based models which are limited by out-of-vocabulary
words. 

Finally, we provide a look into the prediction errors of all models on the SST2 test set.

## Datasets

Datasets splits are as follows:

|              	| Train 	| Validation 	| Test 	|
|:------------:	|:-----:	|:----------:	|:----:	|
|     SST2     	| 63981 	|    3368    	|  872 	|
| Sentiment140 	| 63360 	|    16000   	|  359 	|


Notice that we did not took the whole Sentiment140 dataset as it was too costly to train models on ot.

## Finetuned models

Finetuned models both for SST2 and Sentiment140 were trained with the following parameters:

|             	| Batch size 	| Learning Rate 	| Weigh decay 	| Nb of epochs 	| Number of training examples 	| Number of validation examples 	| Lr scheduler 	| Warmup ratio 	|
|-------------	|------------	|---------------	|-------------	|--------------	|-----------------------------	|-------------------------------	|--------------	|--------------	|
| RoBERTa     	| 12         	| 2e-5          	| 1e-2        	| 5            	| 63981                       	| 872                           	| linear       	| 0.1          	|
| BERT        	| 12         	| 2e-5          	| 1e-2        	| 5            	| 63981                       	| 872                           	| linear       	| 0.1          	|
| DistilBERT  	| 12         	| 2e-5          	| 1e-2        	| 5            	| 63981                       	| 872                           	| linear       	| 0.1          	|
| mBERT       	| 12         	| 2e-5          	| 1e-2        	| 5            	| 63981                       	| 872                           	| linear       	| 0.1          	|
| XLM-ROBERTA 	| 12         	| 2e-5          	| 1e-2        	| 5            	| 63981                       	| 872                           	| linear       	| 0.1          	|
| CANINE-c    	| 6          	| 2e-5          	| 1e-2        	| 5            	| 63981                       	| 872                           	| linear       	| 0.1          	|
| CANINE-s    	| 6          	| 2e-5          	| 1e-2        	| 5            	| 63981                       	| 872                           	| linear       	| 0.1          	|

## Results \& Observations

### Sentiment Classification on benchmark SST2 dataset

|   Accuracy  	| Val set 	| Test set 	|
|:-----------:	|:-------:	|:--------:	|
|     BERT    	|   0.94  	|   0.93   	|
|   RoBERTa   	|   0.94  	|   0.94   	|
|  DistilBERT 	|   0.94  	|   0.91   	|
|    mBERT    	|   0.93  	|   0.88   	|
| XLM-RoBERTa 	|   0.92  	|   0.92   	|
|   CANINE-C  	|   0.93  	|   0.86   	|
|   CANINE-S  	|   0.92  	|   0.85   	|

In this setting, both CANINE-S and CANINE-C perform decently well on the validation set but not as much as the test set.
There are 8 percentage points of difference between CANINE-C and RoBERTa for instance.

### Robustness to noise

in this experience, the goal is to evaluate the models' robustness of noise. To do so, we created 3 noisy versions of the
SST2 dataset where the sentences have been artificially enhanced with noisy (in our case we chose ``KeyboardAug``
from ``nlpaug`` library but in our package 4 other types of noise have been developed - refer to `noisifier/noisifier.py`).

Three levels of noise were chosen: 10\%, 20\% and 40\% . Each word
gets transformed with probability $p$ into a misspelled version of it (see [nlpaug documentation](https://github.com/makcedward/nlpaug/blob/master/nlpaug/augmenter/char/random.py)
for more information).

The noise is **only** applied to the SST2 validation and test sets made of 3368 and 872 examples respectively. 
We compared the 7 models we finetuned on the clean version of SST2 (first experiment) on these 3 noisy datasets (on for 
each level of $p$). The following table gathers the results (averaged over 3 runs):

### Sentiment Classification on more challenging Sentiment140 dataset (tweets)

### Zero-shot transfer learning and domain adaptation from SST2 to Sentiment140

### Analysis of prediction errors on SST2 dataset


