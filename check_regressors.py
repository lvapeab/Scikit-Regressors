from sknn.backend import lasagne
from sknn.mlp import Layer
from sknn.mlp import Regressor as MLP_Regressor
import numpy as np
from sklearn import linear_model, cross_validation, ensemble
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import NuSVR

import argparse
import cPickle
import logging
import math

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%d/%m/%Y_%H:%M:%S: ')
logger = logging.getLogger(__name__)
rng = np.random.RandomState(1)



def get_regressors ():
        return [
            MLP_Regressor(
                    layers=[
                        Layer('Rectifier', units=200),
                        Layer('Rectifier', units=25),
                        Layer('Linear')],
                    learning_rate=0.001,
                    n_iter=500),
            linear_model.SGDRegressor(),
            DecisionTreeRegressor(max_depth=4),
            AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                              n_estimators=500, random_state=rng),
            ensemble.GradientBoostingRegressor(n_estimators=1000,
                                               max_depth = 6,
                                               min_samples_split =1,
                                               learning_rate =0.01,
                                               loss= 'ls'),
        ]

def test_regressor(clf, X_train, y_train, X_dev, y_dev):

    best_model = None

    clf.fit(X_train, y_train)
    logger.info('Results of %s:'%clf.__name__)

    # The mean square error
    logger.info('\tResidual sum of squares: %.2f'
          % np.mean((clf.predict(X_dev).flatten() - y_dev) ** 2))
    # Explained variance score: 1 is perfect prediction
    logger.info('Variance score: %.2f' % clf.score(X_dev, y_dev))

    # predicted = cross_validation.cross_val_predict(clf, X_dev, y_dev, cv=10)
    predicted = clf.predict(X_dev)
    predicted = map(int, predicted)
    acc = accuracy_score(y_dev, predicted)
    mse = mean_squared_error(y_dev, predicted)
    r2 = r2_score(y_dev, predicted, multioutput='variance_weighted')
    logger.info( '\tR2 score: %.4f', r2)
    logger.info( '\tAccuracy: %.4f', acc)
    logger.info( '\tMSE: %.4f', mse)
    return r2




def main(train_data, dev_data, save_path):

    best_model = -1
    best_r2 = -999

    X_train = train_data['embeddings']
    y_train = train_data['classes']

    X_dev = dev_data['embeddings']
    y_dev = dev_data['classes']


    for regressor in get_regressors():
        model, r2 = test_regressor(regressor, X_train, y_train, X_dev, y_dev)
        if math.sqrt(1 - r2**2 ) < best_r2:
            best_r2 = r2
            best_model = regressor

    with open(save_path +'.pkl', 'wb') as f:
        cPickle.dump(best_model, f, protocol=cPickle.HIGHEST_PROTOCOL)


def parse_args():
    parser = argparse.ArgumentParser(
        "Sample (of find with beam-serch) translations from a translation model")
    parser.add_argument("--train",
                        required=True, help="Training data to use")
    parser.add_argument("--dev",
                        required=True, help="Development data to use")
    parser.add_argument("--output", default= 'models/best_accuracy',
                         help="Output path data to use")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    logger.info( 'Loading data')
    train_data = np.load(args.train)
    dev_data = np.load(args.dev)
    save_path = args.output

    main(train_data, dev_data, save_path)
