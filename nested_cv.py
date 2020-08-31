import numbers
import time

import numpy as np
from hyperopt import fmin, Trials, tpe, space_eval
from sklearn import clone
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_validate
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils.multiclass import type_of_target

from utils import tpr_score, fpr_score, measure_sec_of_n_sample_predictions, accuracy_macro_score, \
    one_vs_rest_pr_auc_score


class CustomNestedCV:
    """A general class to handle nested cross-validation for any estimator that
    implements the scikit-learn estimator interface.
    Note: This solution is based on https://github.com/casperbh96/Nested-Cross-Validation library with a couple of
          tweaks & adjustments + support for bayes optimization without support for grid-search

    Parameters
    ----------
    model : estimator
        The estimator implements scikit-learn estimator interface.
    params_grid : dict
        The dict contains hyperparameters for model.
    dataset_name : str
        The Dataset name
    outer_cv : int or cv splitter class (e.g. KFold, StratifiedKFold etc.)
        Outer splitting strategy. If int, StratifiedKFold is default.
    inner_cv : int or cv splitter class (e.g. KFold, StratifiedKFold etc.)
        Inner splitting strategy. If int, StratifiedKFold is default.
    n_jobs : int
        Number of jobs to run in parallel
    random_state : int
        default=None, pass an int for reproducible output across multiple function calls
    cv_options: dict, default = {}
        metric : cv sklearn scorer, default = roc_auc_score
            A scoring metric used to score each model in the inner folds
        randomized_search : boolean, default = False
            Whether to use bayes search from hyperopt or randomized search from sklearn
        search_iter : int, default = 50
            Number of iterations for randomized search or bayesian search
        predict_proba : boolean, default = True
            If true, predict probabilities instead for a class, instead of predicting a class. Used for inner fold metric
        multiclass_average : string, default = 'macro'
            For some classification metrics with a multiclass prediction, you need to specify an
            average other than 'binary'
        multiclass_method : string, default = 'ovr'
            For some classification metrics with a multiclass attribute, you need to specify an
            method such as 'ovr', 'ovo' etc.
        verbose : int, default = 0
            Controls the verbosity: the higher, the more messages.
    wrap_model_using_ovr : boolean, default = False
        If true, wrap given model using One-Vs-rest classifier from sklearn.
    """

    def __init__(self, model, params_grid, dataset_name, outer_cv=10, inner_cv=3, n_jobs=1, random_state=None,
                 cv_options=None, wrap_model_using_ovr=False):
        if cv_options is None:
            cv_options = {}

        self.model = model
        self.params_grid = params_grid
        self.dataset_name = dataset_name
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.wrap_model_using_ovr = wrap_model_using_ovr
        self.scorer = cv_options.get('scorer', None)
        self.randomized_search = cv_options.get('randomized_search', True)
        self.search_iter = cv_options.get('search_iter', 50)
        self.predict_proba = cv_options.get('predict_proba', True)
        self.multiclass_average = cv_options.get('multiclass_average', 'macro')
        self.multiclass_method = cv_options.get('multiclass_method', 'ovr')
        self.verbose = cv_options.get('verbose', 0)

        self.outer_scores = []
        self.best_inner_params_list = []

        self.model.set_params(**{'random_state': self.random_state})

        if self.multiclass_average is not 'macro':
            raise NotImplementedError("Only 'macro' option is supported")

    def _predict_and_score_outer(self, X_test, y_test, outer_model):
        """A method for generating various metrics on current outer fold with best model found based on inner cv"""

        y_pred = outer_model.predict(X_test)
        y_pred_proba = outer_model.predict_proba(X_test)

        tpr = tpr_score(y_test.to_numpy().ravel(), y_pred)
        fpr = fpr_score(y_test.to_numpy().ravel(), y_pred)
        pr_auc = one_vs_rest_pr_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_macro_score(y_test.to_numpy().ravel(), y_pred)
        precision = precision_score(y_test, y_pred, average=self.multiclass_average)
        recall = recall_score(y_test, y_pred, average=self.multiclass_average)

        if self.y_type in 'binary':
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_pred_proba, average=self.multiclass_average,
                                    multi_class=self.multiclass_method)

        return accuracy, tpr, fpr, precision, recall, roc_auc, pr_auc

    def fit(self, X, y):
        """A method to fit nested cross-validation

        Parameters
        ----------
        X : pandas dataframe (rows, columns)
            Training dataframe, where rows is total number of observations and columns
            is total number of features
        y : pandas dataframe
            Output dataframe, also called output variable. y is what you want to predict.
        Returns
        -------
        It will not return directly the values, but it's accessible from the class object it self.
        You should be able to access:
        outer_scores
            Outer scores List.
        best_inner_params_list
            Best inner params for each outer loop as an array of dictionaries
        """

        print(f"\nDataset '{self.dataset_name}'")
        print('{} <-- Running this model now'.format(type(self.model).__name__))

        self.X = X
        self.y = y
        self.y_type = type_of_target(y)

        if isinstance(self.outer_cv, numbers.Number) and isinstance(self.inner_cv, numbers.Number):
            outer_cv = StratifiedKFold(n_splits=self.outer_cv, shuffle=True, random_state=self.random_state)
            inner_cv = StratifiedKFold(n_splits=self.inner_cv, shuffle=True, random_state=self.random_state)
        else:
            outer_cv = self.outer_cv
            inner_cv = self.inner_cv

        outer_scores = []
        best_inner_params_list = []

        # Split X and y into K-partitions to Outer CV
        for (i, (train_index, test_index)) in enumerate(outer_cv.split(X, y)):
            print('\n{}/{} <-- Current outer fold'.format(i+1, self.outer_cv))
            X_train_outer, X_test_outer = X.iloc[train_index], X.iloc[test_index]
            y_train_outer, y_test_outer = y.iloc[train_index], y.iloc[test_index]

            if self.y_type in 'binary':
                auc_scorer = self.scorer if self.scorer is not None else make_scorer(roc_auc_score)
            else:
                auc_scorer = self.scorer if self.scorer is not None else make_scorer(roc_auc_score,
                                                                                     average=self.multiclass_average,
                                                                                     needs_proba=self.predict_proba,
                                                                                     multi_class=self.multiclass_method)

            if self.randomized_search:
                randomized_search_inner_model = clone(self.model)
                search_model = randomized_search_inner_model

                if self.wrap_model_using_ovr:
                    one_vs_rest_randomized_inner_classifier = OneVsRestClassifier(randomized_search_inner_model)
                    search_model = one_vs_rest_randomized_inner_classifier

                randomized_search_cv = RandomizedSearchCV(search_model,
                                                          param_distributions=self.params_grid,
                                                          scoring=auc_scorer,
                                                          cv=inner_cv,
                                                          n_iter=self.search_iter,
                                                          verbose=self.verbose,
                                                          n_jobs=self.n_jobs,
                                                          random_state=self.random_state)
                randomized_search_cv.fit(X_train_outer, y_train_outer)

                best_inner_params = randomized_search_cv.best_params_
            else:
                def objective(params, cv=inner_cv, X_inner=X_train_outer, y_inner=y_train_outer, scorer=auc_scorer):
                    bayes_search_inner_model = clone(self.model)
                    bayes_search_inner_model.set_params(**params)
                    bayes_search_inner_model = clone(bayes_search_inner_model)
                    search_model = bayes_search_inner_model

                    if self.wrap_model_using_ovr and self.y_type not in 'binary':
                        search_model = OneVsRestClassifier(bayes_search_inner_model)

                    cv_search = cross_validate(search_model, X_inner, y_inner, scoring=scorer, cv=cv,
                                               n_jobs=self.n_jobs, verbose=self.verbose)
                    cv_test_score = cv_search['test_score'].mean()

                    # should minimize
                    loss = 1 - cv_test_score

                    if self.verbose > 1:
                        print("Bayesian search iteration params: {}".format(params))
                        print("Bayesian search iteration loss: {}".format(loss))
                        print()

                    return loss

                trials = Trials()
                best_inner_params = fmin(fn=objective, space=self.params_grid, max_evals=self.search_iter,
                                         rstate=np.random.RandomState(self.random_state),
                                         algo=tpe.suggest, trials=trials, verbose=self.verbose)

            if not self.randomized_search:
                best_inner_params = space_eval(self.params_grid, best_inner_params)

            best_inner_params_list.append(best_inner_params)

            # Fit the best hyper-parameters from one of the self.search_iter inner loops
            cloned_model = clone(self.model)
            outer_model = cloned_model

            if self.y_type not in 'binary':
                if not self.randomized_search:
                    cloned_model.set_params(**best_inner_params)
                    cloned_model = clone(cloned_model)
                    one_vs_rest_outer_classifier = OneVsRestClassifier(cloned_model)
                else:
                    one_vs_rest_outer_classifier = OneVsRestClassifier(cloned_model)
                    one_vs_rest_outer_classifier.set_params(**best_inner_params)

                outer_model = one_vs_rest_outer_classifier
            else:
                outer_model.set_params(**best_inner_params)
                outer_model = clone(outer_model)

            train_start = time.time()
            outer_model.fit(X_train_outer, y_train_outer)
            train_stop = time.time()

            training_time_sec = train_stop - train_start

            inference_time_sec = measure_sec_of_n_sample_predictions(outer_model, X_train_outer)

            # Get score and prediction
            accuracy, tpr, fpr, precision, recall, roc_auc, pr_auc = \
                self._predict_and_score_outer(X_test_outer, y_test_outer, outer_model)

            assert tpr == recall

            outer_scores.append([accuracy, tpr, fpr, precision, roc_auc, pr_auc, training_time_sec, inference_time_sec])

            print('Inner fold best parameters: {}'.format(best_inner_params_list[i]))
            print('Results for outer fold:')
            print('Accuracy {}, TPR {}, FPR {}, Precision {}, AUC {}, PR-Curve {}, Training Time {}, Inference Time {}'.format(*outer_scores[i]))

        print()
        print("-" * 100)

        self.outer_scores = outer_scores
        self.best_inner_params_list = best_inner_params_list
