import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def read_data(file_name):
    '''Creates random training and test set from .csv file
    Takes last column as a label

    Parameters
    ----------
    file_name (string) : name of the file with .csv extension

    Returns
    -------
    x_train (pd.DataFrame) :  training data
    x_test (pd.DataFrame) : test data
    y_train (pd.Series) : training labels
    y_test (pd.Series) : traning label
    
    '''
    df = pd.read_csv(file_name)
    df = df.dropna()
    x_column_names = df.columns.values[:-1]
    y_column_name = df.columns.values[-1]
    
    X = df[x_column_names]
    Y = df[y_column_name]
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1/3, 
                                       random_state=1)
    return x_train, x_test, y_train, y_test
    

def calc_distance(a, b):
    '''Calculates distance between two vectors.
    
    Parameters:
    a (pd.Series)               : First vector.
    b (pd.DataFrame, pd.Series) : Second vector.

    Returns:
    distance (float)
    '''
    if isinstance(b, pd.DataFrame):
        b = b.iloc[0]

    if isinstance(a, pd.DataFrame):
        a = a.iloc[0]
        
    distance = np.sqrt(sum((a - b) ** 2))
    return distance

def predict(closest_distances, choice):
    '''Prediction
    

    Parameters
    ----------
    closest_distances (pd.DataFrame) : label in 1st column, distance in 2nd
    choice (string)                  : chosen choice function

    Returns
    -------
    prediction (string) : predicted value

    '''
    label_column = closest_distances.columns.values[0]
    if choice == 'dominant':
        prediction = closest_distances[label_column].value_counts().nlargest(n=1).index[0]
    else:
        print("inna")
        
    return prediction
    
def knn_algorithm(x, y, n, query, choice_function='dominant'):
    '''
    Parameters
    ----------
    x (pd.DataFrame) : training examples
    y (pd.Series)    : prediction examples
    n (int)          : k for KNN
    query (pd.Series): set to predict 
    choice_funtion (string)  : type of choice function for algorithm
    Returns
    -------
    prediction (string)  : predicted value

    '''

    distances = []
    # TODO speed iterations
    for index, row in x.iterrows():
        distances.append(calc_distance(row, query))
    
    y = pd.DataFrame(y)    
    y.loc[:, 'distance'] = distances

    closest_distances = y.nsmallest(n, 'distance')

    available_choice_functions = ['dominant']
    
    if choice_function in available_choice_functions:
        prediction = predict(closest_distances, choice_function)
    else:
        prediction = 'none'
    
    return prediction



x_train, x_test, y_train, y_test = read_data('iris.data.csv')

pred = knn_algorithm(x_train, y_train, 4, x_test.iloc[0], 
                     choice_function = 'dominant')
print("\nPrediction for given query: {}".format(pred))
