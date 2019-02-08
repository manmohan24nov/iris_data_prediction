import matplotlib.pyplot as plt
import import_data
import pandas as pd

raw_data = import_data.main()
print(raw_data.columns)
# raw_data['class'] = raw_data['class'].map({'Iris-setosa': 1, 'Iris-versicolor' : 2, 'Iris-virginica' : 3})

print(raw_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].describe())
print(raw_data.groupby('class').size())

# correlation
print(raw_data.iloc[:,2:].corr())
print(raw_data.iloc[:,0:2].corr())

print(raw_data[raw_data['class'] =='Iris-setosa'].corr())
print(raw_data[raw_data['class'] =='Iris-versicolor'].corr())
print(raw_data[raw_data['class'] =='Iris-virginica'].corr())
print(raw_data[raw_data['class'] =='Iris-virginica'].corr(method='pearson').style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1))

# box and whisker plots
# raw_data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)

# plt.scatter(raw_data['class'], raw_data['sepal_length'],c='r')
# plt.scatter(raw_data['class'], raw_data['sepal_width'],c='b')
# plt.scatter(raw_data['class'], raw_data['petal_length'],c='k')
# plt.scatter(raw_data['class'], raw_data['petal_width'],c='y')
#
# # plt.legend()
plt.show()
