import warnings

from hyperopt import hp
from hyperopt.pyll import scope
from sklearn.tree import DecisionTreeClassifier

from rotation_forest import RotationForestClassifier
from rotboost import RotBoostClassifier
from nested_cv import CustomNestedCV
from utils import *

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
random_state = 0

models = [
    ('RotationForest', RotationForestClassifier(random_state=random_state, n_jobs=-1)),
    ('RotBoost', RotBoostClassifier(DecisionTreeClassifier(random_state=random_state), random_state=random_state, n_jobs=-1))
]

param_grids = {
    'RotationForest': {
        'max_depth': hp.choice('max_depth', [None, scope.int(hp.choice('max_depth_int', np.arange(2, 200, 5, dtype=int)))]),
        'n_estimators': scope.int(hp.quniform('n_estimators', 1, 100, 1)),
        'n_features_per_subset': scope.int(hp.quniform('n_features_per_subset', 2, 6, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 10, 1))
    },
     'RotBoost': {
         'n_estimators': scope.int(hp.quniform('n_estimators', 1, 10, 1)),
         'n_estimators_of_adaboost': scope.int(hp.quniform('n_estimators_of_adaboost', 1, 10, 1)),
         'n_features_per_subset': scope.int(hp.quniform('n_features_per_subset', 2, 6, 1)),
         'adaboost_learning_rate': hp.uniform('adaboost_learning_rate', 0.01, 1),
         'adaboost_base_estimator__max_depth': hp.choice('adaboost_base_estimator__max_depth', np.arange(1, 10, dtype=int))
     }
}

def run_all_datasets(datasets_path: str,
                     models: list,
                     param_grids: dict,
                     random_state: int = 0,
                     randomized_search: bool = True,
                     n_jobs: int = 1,
                     verbose: int = 0,
                     delete_experiments_results_csv_file: bool = False):
    """
    Run over all datasets in the given path and generate a nested cross-validation results.
    All experiments results are saved into experiments_results.csv.
    """

    experiments_results_csv_file_name = "experiments_results.csv"
    if delete_experiments_results_csv_file and os.path.exists(experiments_results_csv_file_name):
        os.remove(experiments_results_csv_file_name)

    all_experiments_df = pd.DataFrame(columns=['Dataset Name', 'Algorithm Name', 'Cross Validation [1-10]', 'Hyper-Parameters Values', 'Accuracy', 'TPR', 'FPR', 'Precision', 'AUC', 'PR-Curve', 'Training Time', 'Inference Time'])

    if not os.path.exists(experiments_results_csv_file_name):
        all_experiments_df.to_csv("experiments_results.csv", index=False)

    for i, transformed_dataset_filename in enumerate(os.listdir(datasets_path)):
        print("Running data-set {}/150".format(i + 1))

        # save dataset name without csv/pkl suffix
        dataset_name = transformed_dataset_filename.partition(".")[0]

        # load data
        df = pd.read_pickle(datasets_path + '/' + transformed_dataset_filename)

        # Split training into X & y convention
        target_column_name = df.columns[-1]
        X, y = df.drop([target_column_name], axis=1), df.iloc[:, -1]
        X = X.copy()
        y = y.copy()

        # run an experiment on the current dataset
        df_experiment = run_experiment(X, y, dataset_name, models, param_grids, random_state,
                                       randomized_search, n_jobs, verbose)

        # append to general experiments dataframe
        df_experiment.to_csv('experiments_results.csv', mode='a', header=False, index=False)


def run_experiment(X: pd.DataFrame,
                   y: pd.DataFrame,
                   dataset_name: str,
                   models: list,
                   param_grids: dict,
                   random_state: int = 0,
                   randomized_search: bool = True,
                   n_jobs: int = 1,
                   verbose: int = 0) -> pd.DataFrame:
    """Perform a nested cross-validation on the given dataset and generate dataframe with the results per outer fold"""

    results = []
    metrics = ['Accuracy', 'TPR', 'FPR', 'Precision', 'AUC', 'PR-Curve', 'Training Time', 'Inference Time']

    # iterate over each model and run nested cross-validation (10 outer, 3 inner)
    for model_name, model in models:
        wrap_using_ovr = True if model_name == 'RotBoost' else False
        nested_cv = CustomNestedCV(model, param_grids[model_name], dataset_name, random_state=random_state, wrap_model_using_ovr=wrap_using_ovr,
                                   n_jobs=n_jobs, cv_options={'verbose': verbose, 'randomized_search': randomized_search, 'search_iter': 50})
        nested_cv.fit(X, y)

        nested_cv_outer_scores = nested_cv.outer_scores
        nested_cv_best_inner_params_list = nested_cv.best_inner_params_list

        for i in range(len(nested_cv_outer_scores)):
            results.append([dataset_name, model_name, str(i+1),
                            dict((remove_prefix(k, 'estimator__'), v) for k, v in nested_cv_best_inner_params_list[i].items())] + nested_cv_outer_scores[i])

    df_result = pd.DataFrame(results, columns=['Dataset Name', 'Algorithm Name', 'Cross Validation [1-10]', 'Hyper-Parameters Values'] + metrics)

    return df_result

#preprocessing_all_datasets('classification_datasets', 'classification_datasets_transformed')
run_all_datasets('classification_datasets_transformed', models, param_grids, randomized_search=False, verbose=1, n_jobs=-1, delete_experiments_results_csv_file=True)
