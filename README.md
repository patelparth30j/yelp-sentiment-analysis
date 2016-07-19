# Synopsis
Our Analysis of [Yelp Data Set](https://www.yelp.com/dataset_challenge) to predict user sentiments based on their review. 

## [Data Cleaning](yelp_01dataCleaning.ipynb)
1.	Lowercase
2.	Remove numbers
3.	Remove stop words using nltk
4.	Porter Stemming
5.	Create sparse matrix representation using scikit.

## [Exploratory Data Analysis](yelp_02EDA.ipynb)
1.	Frequency vs Rank for a sample of yelp review dataset
2.	To find out the stop words we are using inverse term document frequency. 
3.	To create a baseline for evaluating the algorithm, we are plotted the distribution of star category ratings.
4.	To get a better intuition of the text data we plotted the most common and recurring words in each of the reviews.

## Analysis
1.	[Bag of Words Generation](yelp_03bagOfWords.ipynb) - [Bag of words](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) representation of the user reviews.
2.	[Word Embeddings](yelp_04word2vec)- [Word embeddings](https://radimrehurek.com/gensim/models/word2vec.html) representation of the user reviews.
3.	Create models to predict sentiments based on user review and rating
	
	1. [Support Vector Machine](yelp_06SVM.ipynb)
	
	2. [Long Short Term Memory Neural Network](yelp_06LSTM.ipynb)


# Installation
1. Clone the repository

	```
	git clone https://github.com/hrushikesh-dhumal/Yelp-Data-Challlenge.git
	```

2. Dependencies

Install the requirements using `pip install -r requirements.txt`

It is suggested that you have [Anaconda](https://www.continuum.io/downloads) which covers majority of the dependencies. 

# Example
The entire work is in form of python notebook. Execute the playbooks in order of their serial number.

# Author Information
Hrushikesh Dhumal(hrushikesh.dhumal@gmail.com)

Parth Patel(patelparth30j@gmail.com)