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
validation_size = 0.4
seed = 40

clf = KNeighborsClassifier()

for i in range(10,1000,20):
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,
                                                                                Y,
                                                                                test_size=validation_size,
                                                                                random_state=i)
    print('seed',i)
    clf.fit(X_train, Y_train)
    y_pred = (clf.predict(X_validation))

    print('Accuracy Score')
    print(accuracy_score(Y_validation, y_pred) * 100)

# from sklearn.datasets import load_iris
# from sklearn.tree import export_graphviz
# iris=load_iris()
# export_graphviz(clf,
# out_file='iris.dot',  feature_names=iris.feature_names,
#                          class_names=iris.target_names,
#                          filled=True, rounded=True,
#                          special_characters=True)


# DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
#            max_features=None, max_leaf_nodes=None,
#            min_impurity_split=1e-07, min_samples_leaf=1,
#            min_samples_split=2, min_weight_fraction_leaf=0.0,
#            presort=False, random_state=None, splitter='best')
# print(raw_data)
# sc = MinMaxScaler(feature_range = (0, 1))
# training_set_scaled = sc.fit_transform(training_set)