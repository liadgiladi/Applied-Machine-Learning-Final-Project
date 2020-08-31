import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble._forest import ForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.utils import resample, gen_batches, check_random_state
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _check_sample_weight
import random


def random_feature_subsets(array, batch_size, random_state=0):
    """ Generate K subsets of the features in X """
    random_state = check_random_state(random_state)
    features = list(range(array.shape[1]))
    random_state.shuffle(features)
    for batch in gen_batches(len(features), batch_size):
        yield features[batch]


class RotBoostAdaBoostClassifier(AdaBoostClassifier):
    def __init__(self,
                 base_estimator=None,
                 n_estimators=30,
                 learning_rate=1.,
                 random_state=None):

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

        super(RotBoostAdaBoostClassifier, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm='SAMME',  # our version of adaboost is compatible for binary classification,
                                # for multi-class wrap using OneVsRestClassifier or OneVsOneClassifier
            random_state=random_state)

    def _boost(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the adaboost showed in the RotBoost algorithm, not sklearn SAMME"""
        estimator = self._make_estimator(random_state=random_state)

        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict = estimator.predict(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        # if classification is perfect then incorporate into the ensemble with
        # estimator_error=10^-10 instead of stopping
        if estimator_error <= 0:
            estimator_error = pow(10, -10)

        n_classes = self.n_classes_

        # in order to prevent early-stopping when has estimator_error bigger than random classifier, we are going to
        # initialize the sample_weight and try again
        if estimator_error >= 1. - (1. / n_classes):
            self.estimators_.pop(-1)

            sample_weight = _check_sample_weight(None, X, np.float64)
            sample_weight /= sample_weight.sum()
            if np.any(sample_weight < 0):
                raise ValueError("sample_weight cannot contain negative weights")

            return self._boost(iboost, X, y, sample_weight, random_state)

        # Boost weight
        estimator_weight = self.learning_rate * (0.5 * np.log((1. - estimator_error) / estimator_error))

        # Only boost the weights if I will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(estimator_weight * incorrect *
                                    (sample_weight > 0))

        return sample_weight, estimator_weight, estimator_error


class RotBoostRotationAdaBoostClassifier(BaseEstimator):
    def __init__(self,
                 adaboost_base_estimator=None,
                 n_estimators_of_adaboost=30,
                 adaboost_learning_rate=1.,
                 n_features_per_subset=3,
                 rotation_algo='pca',
                 random_state=None):

        self.adaboost_base_estimator = adaboost_base_estimator
        self.n_estimators_of_adaboost = n_estimators_of_adaboost
        self.adaboost_learning_rate = adaboost_learning_rate
        self.n_features_per_subset = n_features_per_subset
        self.rotation_algo = rotation_algo
        self.random_state = random_state

        self.adaboost_classifier = RotBoostAdaBoostClassifier(
            base_estimator=adaboost_base_estimator,
            n_estimators=n_estimators_of_adaboost,
            learning_rate=adaboost_learning_rate,
            random_state=random_state)

    def rotate(self, X):
        if not hasattr(self, 'rotation_matrix'):
            raise NotFittedError('The estimator has not been fitted')

        return np.dot(X, self.rotation_matrix)

    def pca_algorithm(self):
        """ Deterimine PCA algorithm to use. """
        if self.rotation_algo == 'randomized':
            return PCA(svd_solver='randomized', random_state=self.random_state)
        elif self.rotation_algo == 'pca':
            return PCA()
        else:
            raise ValueError("`rotation_algo` must be either "
                             "'pca' or 'randomized'.")

    def _fit_rotation_matrix(self, X):
        seed = check_random_state(None)
        n_samples, n_features = X.shape
        self.rotation_matrix = np.zeros((n_features, n_features),
                                        dtype=np.float32)
        for i, subset in enumerate(
                random_feature_subsets(X, self.n_features_per_subset,
                                       random_state=seed)):
            # take a 75% bootstrap from the rows
            x_sample = resample(X, n_samples=int(n_samples*0.75),
                                random_state=check_random_state(random.randrange(10)*(i+1)))
            pca = self.pca_algorithm()
            pca.fit(x_sample[:, subset])
            self.rotation_matrix[np.ix_(subset, subset)] = pca.components_

    def fit(self, X, y, sample_weight=None, check_input=False):
        self._fit_rotation_matrix(X)

        if type_of_target(y) not in 'binary':
            raise NotImplementedError("RotBoost doesn't support multi-class y")

        self.adaboost_classifier = clone(self.adaboost_classifier)
        self.adaboost_classifier.fit(self.rotate(X), y.reshape(1, -1)[0], sample_weight)

    def predict_proba(self, X, check_input=False):
        return self.adaboost_classifier.predict_proba(self.rotate(X))

    def predict(self, X):
        return self.adaboost_classifier.predict(self.rotate(X))

    def _validate_X_predict(self, X, check_input=False):
        return X


class RotBoostClassifier(ForestClassifier):
    def __init__(self,
                 adaboost_base_estimator=None,  # default base estimator is a DecisionTreeClassifier(max_depth=2)
                 n_estimators=10,
                 n_estimators_of_adaboost=30,
                 n_features_per_subset=3,
                 rotation_algo='pca',
                 adaboost_learning_rate=1.,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
        super(RotBoostClassifier, self).__init__(
            base_estimator=RotBoostRotationAdaBoostClassifier(adaboost_base_estimator, n_estimators_of_adaboost,
                                                              adaboost_learning_rate, n_features_per_subset,
                                                              rotation_algo, random_state),
            estimator_params=("adaboost_base_estimator", "n_estimators_of_adaboost",
                              "adaboost_learning_rate", "n_features_per_subset", "rotation_algo", "random_state"),
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

        self.n_features_per_subset = n_features_per_subset
        self.rotation_algo = rotation_algo
        self.adaboost_base_estimator = adaboost_base_estimator
        self.n_estimators_of_adaboost = n_estimators_of_adaboost
        self.adaboost_learning_rate = adaboost_learning_rate
