# UdacityDataScienceProject2
This is the repository for my work on Project 2 for the Udacity Data Science nanodegree.
Here is a link to this repository: 

This project analyzes disaster data (tweets) from 'Appen' (formally Figure 8) to build a model for an API that classifies these disaster messages.
The tweets are real, and the general idea is to categorize these events so that the appropriate disaster relief agency can investigate accordingly.

The first file, "process_data.py", is used to take in the tweets and apply some natural language processing on them so that they can be used in a machine learning algorithm. Tokenization and Term Frequency-Inverse Document Frequency (TF-IDF) are performed on these tweets.

The second file, "train_classifier.py", is used to take the cleaned corpus of tweets and run them through a classifier. They are then classified into one of 36 different categories, such as "electricity", "fire", or "cold" to be used by the appropriate disaster agency.

The third file, "run.py" creates a web app. The web app has two components- first, there is a search bar where a user can input a tweet and have it be classified. Secondly, there are some visuals that describe the frequencies of the various tweets in the corpus.

This project brings together data engineering skills (process_data.py), builds a machine learning pipeline (train_classifier.py) and then puts the information in a digestible format on the internet (run.py).

##Instructions

First, you will need to clone this repository down to your local machine. 
Seconly, you will need to create a virtual environment. You can do this by running the following command in the command line: `python create virtualmachine' 
