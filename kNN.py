import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import sklearn.neighbors

# reading in the data
customer_data= pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')

# data visualization
customer_data.head() 

X= customer_data[ ['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed',
       'employ', 'retire', 'gender', 'reside'] ]
y= customer_data['custcat']

# data normalization
X= sklearn.preprocessing.StandardScaler().fit_transform(X)

# data train test split
X_train, X_test, y_train, y_test= sklearn.model_selection.train_test_split(X, y, train_size= 0.8, random_state= 1)

# data classification
k= 1 # start with a random value of k
data_fit= sklearn.neighbors.KNeighborsClassifier(n_neighbors= k).fit(X_train, y_train)

# data prediction
y_predicted= data_fit.predict(X_test)

def get_accuaracy(k, X_train, X_test, y_train, y_test):
    model= sklearn.neighbors.KNeighborsClassifier(n_neighbors= k)
    model.fit(X_train, y_train)
    pred_value= model.predict(X_test)
    accuracy= sklearn.metrics.accuracy_score(y_test, pred_value)
    return (accuracy)

acc_= []
while k< 12:
    output= get_accuaracy(k, X_train, X_test, y_train, y_test)
    acc_.append(output)
    k= k + 1

print( 100*max( acc_) )
    







