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
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messages_classified', engine)
    X = df["message"]
    Y = df[df.columns[5:]].astype(int)
    text = X.values.tolist()
    print("-----------------Loading Data Succeeded--------------------")
    return X, Y, df
    pass


def tokenize(text):
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
    print(classification_report(Y_true, Y_pred, target_names=target_names))
    print("-----------------Evaluating Model Succeeded--------------------")
    pass


def save_model(model, model_filepath):
# save
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