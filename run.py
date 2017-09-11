# This is from https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

# Load libraries
import pandas
from pandas.tools.plotting import scatter_matrix
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

from sklearn import preprocessing

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# shape
print(dataset.shape)

# head
print(dataset.head(20))

# descriptions
print(dataset.describe())

# class distribution
print(dataset.groupby('class').size())

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()

# histograms
dataset.hist()
#plt.show()

# scatter plot matrix
scatter_matrix(dataset)
#plt.show()

# Split-out validation dataset
array = dataset.values
# astype allows to overcome the following harmless (?) warning:
# /home/mike/anaconda2/lib/python2.7/site-packages/sklearn/utils/validation.py:429: DataConversionWarning: Data with input dtype object was converted to float64 by StandardScaler.
#  warnings.warn(msg, _DataConversionWarning)

X = array[:,0:4].astype('float64')
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# Mark says:
# decision trees dont need to be scaled.
# logistic regression doesnet either in theory.
# though if you have an exponential type of replationship it can help
# SVMs and neural networks do,as they work on gaussian distributions
# techically you dont even NEED to scale data for them either,but if you dont outliers will vastly overcontribute to the model.
# you can play around with an SVM and the iris dataset,try it scaled and unscaled.

train_scaler = preprocessing.StandardScaler()
test_scaler = preprocessing.StandardScaler()

scaled = True
if scaled:
   X_train = train_scaler.fit_transform(X_train)
   X_validation = test_scaler.fit_transform(X_validation)


# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
print ('Scaled:%s'%scaled)
for name, model in models:
   cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=4, scoring=scoring)
   model.fit(X_train,Y_train)
   print('Model:',name)
   print('Cross Val results:%s'%cv_results.mean())
   Y_validation_predicted  = model.predict(X_validation)
   print ('Out of Sample accuracy:',accuracy_score(Y_validation_predicted, Y_validation))
   print('Confusion Matrix')
   print(confusion_matrix(Y_validation_predicted,Y_validation))
   print('classification_report')
   print (classification_report(Y_validation_predicted,Y_validation))

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

print knn.predict([[5.7,3.0,4.2,1.2]])
