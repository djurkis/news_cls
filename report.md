The scripts are intended to be executed in order. e.g  `python3 train.py; python3 eval.py; python3 classify.py`

The scripts roughly follow the readme instructions, but I decided to keep to a minimal working example without anysort of boilerplate parsing of args to a script


Some notes/thoughs on the task

- Initialy I considered using a mean pooled embedings of the title and article description to represent a document.
- Then train a simple logistic regresssion on top of it which should approximate a more suitable optimization routine. ( label smoothing, extra regularization, tuning .etc)
- However, due to the limmitation of not using any api calls over the network, I didn't have access to any sort of downloads
- ( e.g stop words/tokenizers/parsers , glove embeddings, sent-transformer, etc) For Which I would normally use libraries like Nltk or spacy 
- I decided to use a simpler approach of BOW tf-idf with simple heuristic preprocessing to reduce vocab size and fit a naive bayes classifier, which is a reasonable choice given the limited data regime

- NB approach with tf-idf without any sort of tuning performed poorly on eval set ( I looked at model fit via in/out sample f1 scores ). Altough the model did learn something, as it was significantly stronger than a random baseline.

- After  some clarification  I decided to stick to the original solution and used the pretrained embeddings from sentence-transformers to quickly finish the deliverables.
- I was under the assumption that the task would take me under an hour. which I believe was attainable without the missunderstanding/eda part.

- I spent ~1h on wednesday to do some minor EDA and POC code and ~1h on thursday to finish the deliverables. ~1h on friday to write the report and think about the task in a broader context.


- Given the classification report, There are apparent shortcomings of the apporach.
- I did not use the relationship between similiar categories which could improve the model.
- Possible future low-hanging improvements , during inference use also the confidence of the model to threshold a category, othwerise fallback to a simpler model (tfidf distance ) or even Politics.



