import os
import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from joblib import dump
from pycm import ConfusionMatrix
from sklearn.impute import SimpleImputer
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.utils.multiclass import type_of_target
from sklearn_pandas import gen_features, DataFrameMapper


def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix):]


def find_empty_columns(df: pd.DataFrame) -> list:
    empty_cols = [col for col in df.columns if df[col].isnull().all()]

    return empty_cols


def find_columns_with_missing_values(df: pd.DataFrame) -> list:
    return df.columns[df.isnull().any()].to_list()


def find_columns_that_have_at_most_N_unique_values(df: pd.DataFrame, N, dropna=True) -> list:
    temp_df = df.loc[:, df.nunique(dropna=dropna) <= N]

    if temp_df.empty:
        return []

    return temp_df.columns.to_list()


def find_columns_with_one_value_for_all_rows(df: pd.DataFrame, dropna=True) -> list:
    return find_columns_that_have_at_most_N_unique_values(df, 1, dropna=dropna)


def save_model_into_file(estimator, file_name, path=''):
    dump(estimator, path + file_name + '.joblib')


def apply_numeric_impute_mapper_on_data(input_df: pd.DataFrame, strategy: str = 'mean') -> (pd.DataFrame, DataFrameMapper):
    columns = input_df.select_dtypes(include=np.number).columns.tolist()
    if len(columns) == 0:
        return input_df, None

    transformed_columns = [[col] for col in input_df.select_dtypes(include=np.number).columns.tolist()]

    classes = [{'class': SimpleImputer, 'strategy': strategy, 'fill_value': -9999999}]

    feature_def = gen_features(
        columns=transformed_columns,
        classes=classes
    )

    mapper = DataFrameMapper(feature_def, input_df=True, df_out=True)

    input_df_transformed = mapper.fit_transform(input_df.copy())

    input_df[mapper.transformed_names_] = input_df_transformed

    return input_df, mapper


def apply_categorical_impute_mapper_on_data(input_df: pd.DataFrame, strategy: str = 'constant') -> (pd.DataFrame, DataFrameMapper):
    columns = input_df.select_dtypes("object").columns.tolist()
    if len(columns) == 0:
        return input_df, None

    if strategy == 'most_frequent':
        mapper = DataFrameMapper(
            [([category_feature], SimpleImputer(strategy='most_frequent')) for category_feature in columns],
            input_df=True, df_out=True)
    else:
        transformed_columns = [[col] for col in columns]

        classes = [{'class': SimpleImputer, 'strategy': strategy, 'fill_value': 'missing_value'}]

        feature_def = gen_features(
            columns=transformed_columns,
            classes=classes
        )

        mapper = DataFrameMapper(feature_def, input_df=True, df_out=True)

    input_df_transformed = mapper.fit_transform(input_df.copy())

    input_df[mapper.transformed_names_] = input_df_transformed

    return input_df, mapper


def apply_ohe_mapper_on_data(input_df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([pd.get_dummies(input_df[col], prefix=col) if input_df[col].dtype == object else input_df[col] for col in input_df], axis=1)


def apply_scaler_mapper_on_data(input_df: pd.DataFrame, scaler=StandardScaler) -> (pd.DataFrame, DataFrameMapper):
    columns = input_df.select_dtypes(include=np.number).columns.tolist()
    if len(columns) == 0:
        return input_df, None

    transformed_columns = [[col] for col in columns]

    feature_def = gen_features(
        columns=transformed_columns,
        classes=[scaler]
    )

    mapper = DataFrameMapper(feature_def, input_df=True, df_out=True)

    input_df_transformed = mapper.fit_transform(input_df.copy())

    input_df[mapper.transformed_names_] = input_df_transformed

    return input_df, mapper


def transform_categorical_column_using_label_encoder(df: pd.DataFrame, column: str) -> tuple:
    if df[column].dtype != 'object':
        return df, None

    le = LabelEncoder()

    df[column] = le.fit_transform(df[column])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    return df, le_name_mapping


def preprocessing_dataset(X: pd.DataFrame) -> pd.DataFrame:
    X, _ = apply_numeric_impute_mapper_on_data(X)
    X, _ = apply_scaler_mapper_on_data(X)
    X, _ = apply_categorical_impute_mapper_on_data(X)
    X = apply_ohe_mapper_on_data(X)

    # remove columns with one value
    columns_with_one_value = find_columns_with_one_value_for_all_rows(X)
    if len(columns_with_one_value) > 0:
        X.drop(columns_with_one_value, axis=1, inplace=True)

    return X


def preprocessing_all_datasets(datasets_path: str,
                               transformed_datasets_path: str,
                               y_sampling_min_num: int = 10) -> None:
    """
    Preprocessing all datasets in the given path:
    Removing columns where all samples' value is nan/empty
    Apply 'mean' numeric imputer + standard scaling
    Apply 'constant' categorical imputer + one-hot-encoding
    Perform random over sampler for imbalaned classification data-set
    """
    for dataset_filename in os.listdir(datasets_path):
        print("Preprocessing dataset '{}'".format(dataset_filename))

        dataset_filename_pkl = transformed_datasets_path + '/' + dataset_filename + '.pkl'
        if os.path.exists(dataset_filename_pkl):
            os.remove(dataset_filename_pkl)

        df = pd.read_csv(datasets_path + '/' + dataset_filename)

        if dataset_filename == 'analcatdata_germangss.csv':
            # 'Political_system' is the target column
            df = df[["Age", "Time_of_survey", "Schooling", "Region", "Count", "Political_system"]]

        # we assume the labels are located at the last column
        target_column_name = df.columns[-1]

        df, _ = transform_categorical_column_using_label_encoder(df, target_column_name)

        # split data into X & y convention
        X, y = df.drop([target_column_name], axis=1), df[[target_column_name]]

        # Find the columns where all samples' value is nan/empty
        empty_cols = find_empty_columns(X)

        # Drop empty columns from the dataframe
        if len(empty_cols) > 0:
            print("Dataset '{}', found empty columns [{}]".format(dataset_filename, empty_cols))
            X.drop(empty_cols, axis=1, inplace=True)

        X_transformed = preprocessing_dataset(X)

        # check for imbalanced classification data-set, perform random over sampler
        columns = X_transformed.columns.to_list()
        y_value_counts = y[target_column_name].value_counts().to_dict()
        y_sampling_strategy = {label: count if count >= y_sampling_min_num else y_sampling_min_num for (label, count) in y_value_counts.items()}
        ros = RandomOverSampler(sampling_strategy=y_sampling_strategy, random_state=0)
        X_transformed, y_transformed = ros.fit_resample(X_transformed, y[target_column_name])

        df_transformed = pd.concat([pd.DataFrame(data=X_transformed, columns=columns), pd.DataFrame(data=y_transformed, columns=[target_column_name])], axis=1)

        df_transformed.to_pickle(transformed_datasets_path + '/' + dataset_filename + '.pkl')


def one_vs_rest_pr_auc_score(y_true, y_score):
    """
    Generates the Area Under the Curve for precision and recall in one-vs-rest fashion.
    Also, precision and recall metrics.
    """
    classes = np.unique(y_true)
    n_classes = len(classes)

    if n_classes != y_score.shape[1]:
        raise ValueError(
            "Number of classes in y_true not equal to the number of "
            "columns in 'y_score'")

    y_type = type_of_target(y_true)
    if y_type in 'binary':
        precision, recall, _ = precision_recall_curve(y_true, y_score[:, 1])
        return auc(recall, precision)

    y_true_multilabel = label_binarize(y_true, classes=classes)

    pr_auc_scores = np.zeros((n_classes,))

    for c in range(n_classes):
        y_true_c = y_true_multilabel.take([c], axis=1).ravel()
        y_score_c = y_score.take([c], axis=1).ravel()

        precision_c, recall_c, _ = precision_recall_curve(y_true_c, y_score_c)
        pr_auc_c = auc(recall_c, precision_c)

        pr_auc_scores[c] = pr_auc_c

    return np.average(pr_auc_scores)


def accuracy_macro_score(y_true, y_pred):
    cm = ConfusionMatrix(y_true, y_pred)
    return cm.ACC_Macro


def tpr_score(y_true, y_pred):
    cm = ConfusionMatrix(y_true, y_pred)
    return np.array(list(cm.TPR.values())).mean()


def fpr_score(y_true, y_pred):
    cm = ConfusionMatrix(y_true, y_pred)
    return np.array(list(cm.FPR.values())).mean()


def measure_sec_of_n_sample_predictions(model, X, n=1000):
    samples = X.copy()

    if samples.shape[0] > n:
        samples = samples.loc[:n-1, :]

    number_of_samples = samples.shape[0]

    start = time.time()
    _ = model.predict(samples)
    stop = time.time()

    prediction_time_sec = (stop - start) * (n / number_of_samples)

    return prediction_time_sec
