import pickle
model = pickle.load(open('knn.model', 'rb'))
print model.predict([[5.7,3.0,4.2,1.2]])
