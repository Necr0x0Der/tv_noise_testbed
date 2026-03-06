
from utils.regression import ols_fit_predict
from utils.metrics import r2_score

def evaluate_linear_probe(z_train, y_train, z_test, y_test):
    """ Fits a linear probe via OLS and returns R^2 score along with predictions. """
    y_pred = ols_fit_predict(z_train, y_train, z_test)
    r2 = r2_score(y_test, y_pred)
    return r2, y_pred