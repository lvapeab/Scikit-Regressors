from sknn.backend import lasagne
from sknn.mlp import Regressor, Layer
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.grid_search import RandomizedSearchCV
import numpy as np
from sklearn import linear_model, cross_validation, ensemble
from sklearn.metrics import mean_squared_error, accuracy_score


print "Loading data"

train_data = np.load('/home/lvapeab/PycharmProjects/Scikit-Regressors/data/xerox_train.npz')
dev_data = np.load('/home/lvapeab/PycharmProjects/Scikit-Regressors/data/xerox_test.npz')

X_train = train_data['embeddings']
y_train = train_data['classes']


X_dev = dev_data['embeddings']
y_dev = dev_data['classes']

print "Training SGD Regressor"

clf = linear_model.SGDRegressor()
clf.fit(X_train, y_train)
print("Results of SK Regression....")

# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((clf.predict(X_dev).flatten() - y_dev) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % clf.score(X_dev, y_dev))

predicted = cross_validation.cross_val_predict(clf, X_dev, y_dev, cv=10)
predicted = map(int, predicted)
acc = accuracy_score(y_dev, predicted)
mse = mean_squared_error(y_dev, predicted)

print "Accuracy: ", acc
print "MSE:", mse

print "Training SGD NN Regressor"

nn = Regressor(
    layers=[
        Layer("Rectifier", units=25),
        Layer("Linear")],
    learning_rate=0.0001,
    n_iter=500)


nn.fit(X_train, y_train)
print("Results of SKNN Regression....")

# print('Coefficients: ', regr.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((nn.predict(X_dev).flatten() - y_dev) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % nn.score(X_dev, y_dev))

predicted = cross_validation.cross_val_predict(nn, X_dev, y_dev, cv=10)
predicted = map(int, predicted)
acc = accuracy_score(y_dev, predicted)
mse = mean_squared_error(y_dev, predicted)

print "Accuracy: ", acc
print "MSE:", mse



print "Training Boosting Regressor"


params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)


print("Results of SK Ensemble Regression....")

# print('Coefficients: ', regr.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((clf.predict(X_dev).flatten() - y_dev) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % clf.score(X_dev, y_dev))
predicted = cross_validation.cross_val_predict(clf, X_dev, y_dev, cv=10)
predicted = map(int, predicted)
acc = accuracy_score(y_dev, predicted)
mse = mean_squared_error(y_dev, predicted)

print "Accuracy: ", acc
print "MSE:", mse