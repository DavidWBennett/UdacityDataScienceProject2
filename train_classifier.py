import sys
import pandas as pd
from sqlalchemy import create_engine
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
import pickle

def load_data(database_filepath):
    """
    Input: 
    database_filepath -> The database filepath where the data is stored.
    
    Output:
    X -> The features that will be used to classify the messages.
    Y -> The designated classification for the tweets. This will be used in building a machine learning model.
    df -> The dataframe to be used in the classification model.
    
    This function loads the data to be used in the  machine learning model.
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messages_classified', engine)
    X = df["message"]
    Y = df[df.columns[5:]].astype(int)
    text = X.values.tolist()
    print("-----------------Loading Data Succeeded--------------------")
    return X, Y, df
    pass


def tokenize(text):
    """
    Input: 
    text -> The text (strings) that will be converted into tokens.
    
    Output:
   clean_tokens -> The prepared tokens to be used in a machine learning model.
    
    This function takes a corpus of text and lemmatizes it into clean tokens to be used in a machine learning model.
    """
    tokens = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    #print("-----------------Tokenizing Data Succeeded--------------------")
    return clean_tokens
    pass


def build_model():
    """
    Output:
    pipeline -> The model to be used to classify the tokenized text.
    
    This function prepares a pipeline with a count vectorizer, TF-IDF transformer, and a multi output classifier (using a random forest classifier).
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    

    print("-----------------Building Model Succeeded--------------------")
    return pipeline
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Input:
    model -> The pipeline created in the build_model() function.
    X_test -> The features of the test set to be used for model training.
    Y_test -> The reponse (message categories) to be used for model training.
    category_names -> The list of categories that the text can be classified into.
    
    This function trains the model build in the build_model() function. It then prints a classification report based on the training data, then performs a grid search to 
    determine the optimal parameters. These optimal parameters are then printed out as well.
    """
    predicted = model.predict(X_test)
    Y_true = Y_test.reset_index(drop = True)
    Y_pred = predicted
    target_names = category_names#list(y.columns)
    print(classification_report(Y_true, Y_pred, target_names=target_names))
    parameters = {
        'clf__estimator': [RandomForestClassifier(), AdaBoostClassifier()]
        }

    cv = GridSearchCV(model, param_grid=parameters)
    print(cv)
    cv.fit(X_test, Y_test)
    y_pred = cv.predict(X_test)
    print("\nBest Parameters:", cv.best_params_)
    #print(classification_report(Y_true, Y_pred, target_names=target_names))
    print("-----------------Evaluating Model Succeeded--------------------")
    pass


def save_model(model, model_filepath):
     """
    Input: 
    model -> The optimzed model to be saved.
    model_filepath -> The file path where the pickled model will be saved.
    
    This function takes in a model, pickles (serializes) it, then saves it.
    """
    pickle.dump(model,open(model_filepath,'wb'))
    print("-----------------Saving Model Succeeded--------------------")
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
