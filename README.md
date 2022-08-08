# UdacityDataScienceProject2
This is the repository for my work on Project 2 for the Udacity Data Science nanodegree.
Here is a link to this repository: https://github.com/DavidWBennett/UdacityDataScienceProject2

This project analyzes disaster data (tweets) from 'Appen' (formally Figure 8) to build a model for an API that classifies these disaster messages.
The tweets are real, and the general idea is to categorize these events so that the appropriate disaster relief agency can investigate accordingly.

The first file, "process_data.py", is used to take in the tweets and apply some natural language processing on them so that they can be used in a machine learning algorithm. Tokenization and Term Frequency-Inverse Document Frequency (TF-IDF) are performed on these tweets.

The second file, "train_classifier.py", is used to take the cleaned corpus of tweets and run them through a classifier. They are then classified into one of 36 different categories, such as "electricity", "fire", or "cold" to be used by the appropriate disaster agency.

The third file, "run.py" creates a web app. The web app has two components- first, there is a search bar where a user can input a tweet and have it be classified. Secondly, there are some visuals that describe the frequencies of the various tweets in the corpus.

This project brings together data engineering skills (process_data.py), builds a machine learning pipeline (train_classifier.py) and then puts the information in a digestible format on the internet (run.py).

## Instructions

1. You will need to clone this repository down to your local machine. 
2. You will need to create a virtual environment. You can do this by running the following command in the command line: 
    `python create virtualmachine'
3.  You will run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        
 4. You will run the following command in the app's directory to run the web app.
  -`python run.py`
  
 To view the web app, go to: http://0.0.0.0:3001/
