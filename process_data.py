import sys
import pandas as pd
from sqlalchemy import create_engine
#import sqllite3


def load_data(messages_filepath, categories_filepath):
    """
    Input: 
    messages_filepath -> The file path where the messages datasaet is stored.
    categories_filepath -> The file path where the categories dataset is stored.
    
    Output:
    df -> The concatenation of the messages and categories datasets.
    
    This function takes in messages and categories data and merges them. It then converts the categories column from long format to wide format.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.join(categories, lsuffix = '_message', rsuffix = '_category')
    
    categories = df["categories"].str.split(";", n = -1, expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].astype(float)
    df.drop("categories", axis = 1, inplace = True)
    df = pd.concat([df, categories], axis = 1, sort = True)
    print("-----------------Loading Data Succeeded--------------------")
    return df
    pass


def clean_data(df):
     """
    Input: 
    df -> The dataframe to be cleaned.
    
    Output:
    df -> The cleaned dataframe.
    
    This function takes in a dataframe and cleans it by removing duplicate entries and ensuring that each category is strictly binary.
    """
    df.loc[df['related'] == 2].index #193 rows had a value of '2' when it should be 0 or 1
    df.drop(index = df.loc[df['related'] == 2].index, inplace = True)
    df.related.unique()
    duplicateRows = df[df.duplicated()]
    df.drop_duplicates(inplace = True)
    print("-----------------Cleaning Data Succeeded--------------------")
    return df
    pass


def save_data(df, database_filename):
     """
    Input: 
    df -> The dataframe to be saved.
    database_filename -> The SQL Lite database file name where the dataframe will be saved.
    
    This function takes in a dataframe and saves it to a SQL Lite database table.
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages_classified', engine, if_exists='replace', index=False)
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
