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

Following the previous experiment, we decided to test how CANINE would perform on Sentiment140 without having been train
on it. It would allow us to see how CANINE and other models perform when faced with "natural" noise (language in tweets)
and when the domain is different (in the sense that the topic and the way of writing/speaking are different). Additionally,
we can quantify the gain in accuracy from directly training on Sentiment140 compared to doing zero-shot transfer.

As CANINE has been pre-trained on multilingual data, it could be worth it to analyze its abilities on other languages
than English, especially since it is tokenization-free and hence, theoretically, should be able to adapt more easily to
languages with richer morphology. To test that, we did zero-shot transfer learning on multilingual data (MARC dataset).

To go further, we decided to compare the abilities of CANINE and other BERT-like models when actually finetuned on this
multilingual data. To do so, we have chosen to work again with the MARC dataset, using data in German, Japanese and
Chinese. We would like to see how CANINE compares and if it is better on languages which are more challenging for
token-based models (Chinese for instance). Compared to the previous experience, we are not doing transfer learning but
really finetuning for 2 epochs the models on a train set.

Finally, we provide a look into the prediction errors of all models on the SST2 test set.

## Datasets

Datasets splits are as follows:

|                      	                       | Train 	  | Validation 	 | Test 	 |
|:--------------------------------------------:|:--------:|:------------:|:------:|
|                  SST2     	                  | 63981 	  |  3368    	   | 872 	  |
|                Sentiment140 	                | 63360 	  |  16000   	   | 359 	  |
| Amazon Reviews Multilingual (per language) 	 | 160000 	 |   4000   	   | 4000 	 |


Notice that we did not took the whole Sentiment140 dataset as it was too costly to train models on ot.

## Finetuned models

Finetuned models both for SST2 and Sentiment140 were trained with the following parameters:

|             	| Batch size 	| Learning Rate 	| Weigh decay 	| Nb of epochs 	 | Number of training examples 	| Number of validation examples 	| Lr scheduler 	| Warmup ratio 	|
|-------------	|------------	|---------------	|-------------	|----------------|-----------------------------	|-------------------------------	|--------------	|--------------	|
| RoBERTa     	| 12         	| 2e-5          	| 1e-2        	| 3            	 | 63981                       	| 872                           	| linear       	| 0.1          	|
| BERT        	| 12         	| 2e-5          	| 1e-2        	| 3            	 | 63981                       	| 872                           	| linear       	| 0.1          	|
| DistilBERT  	| 12         	| 2e-5          	| 1e-2        	| 3            	 | 63981                       	| 872                           	| linear       	| 0.1          	|
| mBERT       	| 12         	| 2e-5          	| 1e-2        	| 3            	 | 63981                       	| 872                           	| linear       	| 0.1          	|
| XLM-ROBERTA 	| 12         	| 2e-5          	| 1e-2        	| 3            	 | 63981                       	| 872                           	| linear       	| 0.1          	|
| CANINE-c    	| 6          	| 2e-5          	| 1e-2        	| 3            	 | 63981                       	| 872                           	| linear       	| 0.1          	|
| CANINE-s    	| 6          	| 2e-5          	| 1e-2        	| 3            	 | 63981                       	| 872                           	| linear       	| 0.1          	|

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
There are 8 percentage points of difference between CANINE-C and RoBERTa for instance. mBERT has similar behavior than
the two CANINE models.

### Robustness to noise

In this experience, the goal is to evaluate the models' robustness of noise. To do so, we created 3 noisy versions of the
SST2 dataset where the sentences have been artificially enhanced with noisy (in our case we chose ``RandomCharAug``
from ``nlpaug`` library with action `substitute` but in our package 4 other types of noise have been developed - refer 
to `noisifier/noisifier.py`).

Three levels of noise were chosen: 10\%, 20\% and 40\% . Each word gets transformed with probability $p$ into a misspelled 
version of it (see [nlpaug documentation](https://github.com/makcedward/nlpaug/blob/master/nlpaug/augmenter/char/random.py)
for more information).

The noise is **only** applied to the SST2 validation and test sets made of 3368 and 872 examples respectively. 
We compared the 7 models we finetuned on the clean version of SST2 (first experiment) on these 3 noisy datasets (on for 
each level of $p$). The following table gathers the results (averaged over 3 runs):

|             	| Noise level 10% 	|          	| Noise level 20% 	|          	| Noise level 40% 	|          	|
|:-----------:	|:---------------:	|:--------:	|:---------------:	|:--------:	|:---------------:	|:--------:	|
|             	|     Val set     	| Test set 	|     Val set     	| Test set 	|     Val set     	| Test set 	|
|     BERT    	|       0.88      	|   0.87   	|       0.85      	|   0.82   	|       0.80      	|   0.80   	|
|   RoBERTa   	|       0.88      	|   0.89   	|       0.87      	|   0.85   	|       0.83      	|   0.82   	|
|  DistilBERT 	|       0.85      	|   0.82   	|       0.82      	|   0.79   	|       0.76      	|   0.76   	|
|    mBERT    	|       0.88      	|   0.82   	|       0.85      	|   0.80   	|       0.80      	|   0.76   	|
| XLM-RoBERTa 	|       0.89      	|   0.85   	|       0.86      	|   0.83   	|       0.81      	|   0.81   	|
|   CANINE-C  	|       0.86      	|   0.80   	|       0.83      	|   0.76   	|       0.79      	|   0.74   	|
|   CANINE-S  	|       0.85      	|   0.80   	|       0.83      	|   0.77   	|       0.78      	|   0.74   	|

Both CANINE models have a better performance than DistilBERT for a high level of noise (>= 40\%). However all other models
are better to handle this type of artificial noise, RoBERTa being the best of all. 

### Sentiment Classification on more challenging Sentiment140 dataset (tweets)

The following experience is meant to evaluate the performances of the various models on a more challenging dataset: 
Sentiment140. This dataset is made of 1.6 million of tweets, all in English. The language used is very different from the
one in SST2 as it is made of more abbreviations, colloquialisms, slang, etc. Therefore it is expected to be hard for the
models to handle such text (which is "naturally" noisy). CANINE has a theoretical advantage on such dataset due to the
fact that it is tokenizer-free and operates at the character level.

The following table reports the results we obtained when finetuning all models on the (smaller) training set of 63360 examples.

|                 	| **Val set** 	 | **Test set** 	 |
|:---------------:	|:-------------:|:--------------:|
|     **BERT**    	|   0.84    	   |   0.86     	   |
|   **RoBERTa**   	|   0.87    	   |   0.86     	   |
|  **DistilBERT** 	|   0.83    	   |   0.85     	   |
|    **mBERT**    	|   0.79    	   |   0.78     	   |
| **XLM-RoBERTa** 	|   0.81    	   |   0.80     	   |
|   **CANINE-C**  	|   0.79    	   |   0.78     	   |
|   **CANINE-S**  	|   0.80    	   |   0.79     	   |

### Zero-shot transfer learning and domain adaptation from SST2 to Sentiment140

In this experience we would like to see how CANINE models perform when they are faced with "natural" noise that they were
**not** trained on. Compared to the previous experience where models where trained on Sentiment140, here models are 
trained on SST2 but evaluated on validation and test set from Sentiment140. 

In the previous task, CANINE models were not the best performing one. Actually, with mBERT, they were the last ones. Here 
we are evaluating something different: the ability for a model to adapt to another domain (in the sense that the topic 
and the way of writing/speaking are different) in a zero-shot transfer setting. It might be that, in real life settings, 
one has access to a clean benchmark-type dataset (such as SST2) but wants to do inference on a dataset whose subject is
quite different and full of misspellings and grammar errors. 

Results are reported in the following table:

|                 	| **Val set** 	| **Test set** 	|
|:---------------:	|:-----------:	|:------------:	|
|     **BERT**    	|     0.72    	|     0.84     	|
|   **RoBERTa**   	|     0.73    	|     0.88     	|
|  **DistilBERT** 	|     0.71    	|     0.82     	|
|    **mBERT**    	|     0.68    	|     0.76     	|
| **XLM-RoBERTa** 	|     0.72    	|     0.83     	|
|   **CANINE-C**  	|     0.64    	|     0.77     	|
|   **CANINE-S**  	|     0.64    	|     0.73     	|

CANINE models do not perform well on this task. They have -9 percentage point of accuracy compared to RoBERTa for 
instance (best performing model on this task) on the validation set. We noticed that mBERT has more difficulties than 
other BERT-like models on Sentiment140 dataset overall. Again, CANINE and mBERT have similar behavior.

### Zero-shot transfer learning on multlingual data

This experiment builds on the idea that CANINE is expected to perform better on languages with a different morphology
than English, for instance on non-concatenative morphology (such as Arabic and Hebrew), compounding (such as German and
Japanese), vowel harmony (Finnish), etc. Moreover, it is known that splitting on whitespaces (which is often done in most
tokenizer - note that SentencePiece has an option to skip whitespace splitting) is not adapted to languages such as Thai 
or Chinese. 

In this experience, models have been finetuned on the English dataset SST2 and are only evaluated both on validation and
tests sets of 4 languages from the Multilingual Amazon Reviews Corpus ([MARC](https://arxiv.org/abs/2010.02573)). We 
considered the four following language: German, French, Japanese and Chinese for their morphological properties. 

This dataset contains for each review the number of stars associated by the reviewer. To derive positive/negative
sentiment from this, we considered that if 1 or 2 stars only have been associated to the review, the sentiment is 
negative. While if 4 or 5 stars have been chosen, the review is positive. Neutral reviews, with 3 stars, were not 
considered. For each language, this gives us 160000 training samples, 4000 validation samples and 4000 test samples.

Results are given in the following table:

|                 	|  **French** 	|              	|  **German** 	|              	| **Japanese** 	|              	| **Chinese** 	|              	|
|:---------------:	|:-----------:	|:------------:	|:-----------:	|:------------:	|:------------:	|:------------:	|:-----------:	|:------------:	|
|                 	| **Val set** 	| **Test set** 	| **Val set** 	| **Test set** 	|  **Val set** 	| **Test set** 	| **Val set** 	| **Test set** 	|
|    **mBERT**    	|     0.71    	|     0.70     	|     0.66    	|     0.66     	|     0.56     	|     0.55     	|     0.58    	|     0.59     	|
| **XLM-RoBERTa** 	|     0.87    	|     0.86     	|     0.86    	|     0.87     	|     0.87     	|     0.85     	|     0.80    	|     0.79     	|
|   **CANINE-C**  	|     0.70    	|     0.69     	|     0.59    	|     0.58     	|     0.50     	|     0.50     	|     0.57    	|     0.55     	|
|   **CANINE-S**  	|     0.71    	|     0.70     	|     0.61    	|     0.61     	|     0.52     	|     0.52     	|     0.57    	|     0.57     	|

CANINE-S is similar to mBERT for French and Chinese data. Overall XLM-RoBERTa is extremely better than other models. 
Note that its pre-training strategy is different from the one of mBERT and CANINE. Indeed, while mBERT and CANINE have both been 
pretrained on the top 104 languages with the largest Wikipedia using a MLM objective, XLM-RoBERTa was pretrained on 2.5TB 
of filtered CommonCrawl data containing 100 languages. This might be a confounding variable.

### Finetuning on multlingual data

In this last experiment, we now compare CANINE to other BERT-like models on multilingual data where they are finetuned 
on it. This is the difference with the previous experience. To do so, we have chosen to work again with the MARC dataset, 
using data in German, Japanese and Chinese. We would like to see how CANINE compares and if it is better on languages
which are more challenging for token-based models (Chinese for instance). 

Please note that due to time and compute constraints, we considered only one CANINE model, CANINE-S. 

The results are given below:

|             	|  German 	|          	| Japanese 	|          	| Chinese 	|          	|
|:-----------:	|:-------:	|:--------:	|:--------:	|:--------:	|:-------:	|:--------:	|
|             	| Val set 	| Test set 	|  Val set 	| Test set 	| Val set 	| Test set 	|
|    mBERT    	|   0.93  	|   0.93   	|   0.92   	|   0.92   	|   0.87  	|   0.88   	|
| XLM-RoBERTa 	|   0.92  	|   0.92   	|   0.93   	|   0.93   	|   0.88  	|   0.88   	|
|   CANINE-S  	|   0.93  	|   0.93   	|   0.90   	|   0.89   	|   0.85  	|   0.85   	|

Quite surprisingly, on German, CANINE-S is slightly better than XLM-RoBERTa and has similar performance than mBERT. 
However on Japanese and Chinese, it is not the case. mBERT and especially XLM-RoBERTa should be preferred has they
provide better accuracy on both validation and test sets. 

### Analysis of prediction errors on SST2 dataset
