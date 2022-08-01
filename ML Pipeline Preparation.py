#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[1]:


# import libraries
import pandas as pd
from sqlalchemy import create_engine
#from nltk.tokenize import word_tokenize
#from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


# In[2]:


# load data from database
engine = create_engine('sqlite:///MessageCategories.db')
df = pd.read_sql_table('MessageCategories', engine)

X = df["message"]#[0:500]
y = df[df.columns[5:]].astype(int)#[0:500].astype(int)
#print(y)


# In[3]:


#I think I need this to be a corpus, like this: ["This is a sentence.", "This is a word."]
corpus = X.values.tolist()
print(corpus[2:6])


# ### 2. Write a tokenization function to process your text data

# In[4]:



def tokenize(text):
    #tokens_list = []
    #for i in range(2, 6):
    #    token = nltk.word_tokenize(text)
    #    tokens_list.append(token)
    #return tokens_list
    tokens = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    
    #return tokens
    pass

#X1 = tokenize(corpus)
#CountVectorizer(corpus[2:6], tokenizer = tokenize)


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[5]:


pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier())),
])


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

pipeline.fit(X_train, y_train)
predicted = pipeline.predict(X_test)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[7]:


y_true = y_test.reset_index(drop = True)
y_pred = predicted
target_names = list(y.columns)
print(classification_report(y_true, y_pred, target_names=target_names))


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[9]:


#pipeline2 = Pipeline([
#    ('scaler', StandardScaler()),
#    ('clf', RandomForestClassifier())
#])

parameters = {
    'clf__estimator': [RandomForestClassifier(), AdaBoostClassifier()]
    #'clf__n_jobs': [10, 100]
}


cv = GridSearchCV(pipeline, param_grid=parameters)
print(cv)


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[10]:


cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)


# In[13]:


print("\nBest Parameters:", cv.best_params_)
y_true = y_test.reset_index(drop = True)
target_names = list(y.columns)
print(classification_report(y_true, y_pred, target_names=target_names))


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[ ]:


pipeline2 = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(AdaBoostClassifier())),
])


# ### 9. Export your model as a pickle file

# In[15]:


import pickle

# save
with open('pipeline.pkl','wb') as f:
    pickle.dump(pipeline,f)


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




