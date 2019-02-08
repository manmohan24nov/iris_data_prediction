from sklearn.preprocessing import MinMaxScaler
import import_data
import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

raw_data = import_data.main()
raw_data_array = raw_data.values
X = raw_data_array[:,0:4]
Y = raw_data_array[:,4]
validation_size = 0.2
seed = 555

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,
                                                                                Y,
                                                                                test_size=validation_size,
                                                                                random_state=seed)

scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# print(models)
# evaluate each model in turn
results = []
names = []
for name, model in models:
    # print(name,model)
    for i in range(10,1000,20):
        kfold = model_selection.KFold(n_splits=10, random_state=i)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)