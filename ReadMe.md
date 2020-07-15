
## Implementation of Machine Learning and Deep Learning for Sentiment Analysis
* Sentiment analysis is the process of determining whether a piece of writing is positive, negative or neutral. 
* In this project I have demondtrated how various Machine Learning and Deep Learning models can be used for sentiment analysis.

## The Dataset:
* The dataset used is "Sentiment Labelled Sentences Dataset", from the UC Irvine Machine Learning Repository.
* The sentences come from three different websites/fields:
    * amazon.com
    * imdb.com
    * yelp.com
* Each sentence is labelled as either 1 (for positive) or 0 (for negative).
* For each website,tThere exist 500 positive and 500 negative sentences.
* This dataset was created for the Paper 'From Group to Individual Labels using Deep Features', Kotzias et. al,. KDD 2015.  *(Please cite the paper if you want to use it :))*

* Link to the dataset is: [Sentiment Labelled Sentences Data Set](https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences)
* The dataset is present in the Dataset folder.

## Machine Learning models:
* I have used the follwoing Machine Learning models:

 1. Multinomial Naive bayes
 2. Random Forest
 3. LinearSVC


* The code implementing these models is in 'modules/Sentiment_Analysis_ML.ipynb'.
* All the trained models are stored at 'models/ML'. Thereafter the models are segrated as per the dataset (Amazon, IMDB, Yelp).

## Deep Learning models:
* I have used the follwoing Deep Learning models:

 1. Feed Forward Neural Network (FFNN)
 2. Convolutional Neural Network (CNN)
 3. Recurrent Neural Network (LSTM)


* As the dataset consists of three different set of data, I have created three different implementations for each of them.

 1. Amazon product Rreview Dataset  ('modules/Amazon_Sentiment_Analysis_DL.ipynb')
 2. IMDB Movie Review Dataset  ('modules/IMDB_Sentiment_Analysis_DL.ipynb')
 3. Yelp Restuarant Review Dataset  ('modules/Yelp_Sentiment_Analysis_DL.ipynb')


* All the trained models are stored at 'models/DL'. Thereafter the models are segrated as per the dataset (Amazon, IMDB, Yelp).

### Word Embeddings:
* All the Deep Learning architectures use the GloVe Word Embeddings.
* To download [click here](https://www.kaggle.com/rtatman/glove-global-vectors-for-word-representation?select=glove.6B.100d.txt) (please download them before running the code.)
* The 6 Billion words, 100 dimensional vector representation variant is used.
* The have been stored at location 'Dataset/GloVe_Word_Embeddings'

### Results:
After tyring various machine learning and deep learning models, I got the following results.

|Model|Amazon Reviews|IMDB Reviews|Yelp Reviews|
|:-------|:--------|:-------|:--------|
|Multinomial Naive Bayes|85%|85%|78%|
|Random Forest|80%|79%|79%|
|Linear SVC|84%|81.50%|80%|
|FFNN|81.50%|84%|82%|
|CNN|87%|85.50%|82.50%|
|LSTM|87%|85%|83%|
