import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree

#importing wine data
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')

print("\nThe wine dataset:\n")
print data.head()

#splitting label and features
y = data.quality
X = data.drop('quality', axis=1)

#slpitting into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
print "\nX_train:\n"
print(X_train.head())
print X_train.shape


#preprocessing: making X in range of -1 to 1
X_train_scaled = preprocessing.scale(X_train)
print("\nAfter preprocessing: \n")
print X_train_scaled
print X_train_scaled.shape

#Using Decision Tree Classifier
clf=tree.DecisionTreeClassifier()

#Fitting: Training the ML Algo
clf.fit(X_train, y_train)

#Obtaining the confidence score for SVR
confidence = clf.score(X_test, y_test)
print("\nThe confidence score:\n")
print(confidence)


#predicting the forcasts
y_pred = clf.predict(X_test)

#printing fthe prediction
print("\nThe prediction:\n")
print(y_pred)

#printing the labeled result expectation
print("\nThe expectation:\n")
print(y_test)

#converting the numpy array to list
x=np.array(y_pred).tolist()

#printing first 5 predictions
print("\nThe prediction:\n")
for i in range(0,5):
    print x[i]
    
#printing first five expectations
print("\nThe expectation:\n")
print y_test.head()



