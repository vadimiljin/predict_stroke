# Standard library
from collections import Counter
from typing import List, Dict, Optional, Tuple, Any

# Basic data manipulation and visualization
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from numpy import float64

# Preprocessing libraries
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer

# Machine learning libraries
from sklearn.utils import resample
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    GridSearchCV,
    KFold,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    BaggingClassifier,
)
from sklearn import tree
import xgboost as xg
from catboost import Pool, CatBoostClassifier, cv
from sklearn.inspection import permutation_importance
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.utils import parallel_backend

# Imbalance handling libraries
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import make_pipeline

# Statistical inference libraries
import statsmodels.api as sm
import statsmodels.stats.weightstats as smweight
import statsmodels.stats.proportion as smprop
from scipy.stats import chisquare

# Interpretability libraries
import shap

# Visualization settings
palette = {"Yes": "red", "No": "green"}


def create_boxplot(data: pd.DataFrame, list_of_columns: List[str]) -> plt.figure:
    """Creates boxplots for numerical features of a dataset.

    Boxplots provide a good understanding of how data are spread out in our dataset.

    Args:
        data (pd.DataFrame): Input data.
        list_of_columns (List[str]): The feature we want to plot a boxplot for.

    Returns:
        plt.figure: Subplots of boxplots for each of the features.
    """
    fig, axs = plt.subplots(1, 3, figsize=(16, 6))

    for i, (ax, curve) in enumerate(zip(axs.flat, list_of_columns)):
        sns.boxplot(
            y=data[curve],
            color="darkorange",
            ax=ax,
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "black",
                "markeredgecolor": "black",
                "markersize": "6",
            },
            flierprops={"marker": "o", "markeredgecolor": "darkgreen"},
        )
        ax.set_title(list_of_columns[i])
        ax.set_ylabel("")
    plt.show()


#    return fig


def create_histplot(data: pd.DataFrame, list_of_columns: List[str]) -> plt.figure:
    """Creates histograms for numerical features of a dataset.

    Histograms provide a good understanding of how data are distributed.

    Args:
        data (pd.DataFrame): Input data.
        list_of_columns (List[str]): The features we want to plot a histogram for.

    Returns:
        plt.figure: Subplots of histograms for each of the features.
    """
    fig, axes = plt.subplots(
        1, 3, figsize=(16, 6), gridspec_kw={"hspace": 0.75, "wspace": 0.25}
    )

    for i, ax in enumerate(axes.flatten()):
        sns.histplot(data=data, x=data[list_of_columns[i]], ax=ax, kde=True)
        ax.ticklabel_format(style="plain")
        ax.set_xlabel("")
        ax.set_title(f"{' '.join(list_of_columns[i].split('_'))}", fontsize=13, y=1.03)
        ax.ticklabel_format(style="sci")

    sns.despine(left=True)
    plt.show()


#    return fig


def percentage_plot(data: pd.DataFrame, feature_name: str) -> None:
    """Creates a percentage plot showing the distribution of a feature in relation to stroke occurrence.

    The function calculates the percentages of patients with and without stroke for the given feature
    and plots them as a horizontal bar chart.

    Args:
        data (pd.DataFrame): The input data.
        feature_name (str): The feature for which the distribution is to be plotted.

    """
    percentages = (
        data.groupby([feature_name, "stroke"])
        .agg({"stroke": "count"})
        .rename(columns={"stroke": "nr_of_patients"})
        .groupby(level=0)
        .transform(lambda x: x / x.sum() * 100)
        .reset_index()
    )

    # Sort by percentage of patients
    percentages.sort_values(by="nr_of_patients", ascending=False, inplace=True)

    fig, ax = plt.subplots(figsize=(6, 3))
    for stroke, color in zip(["No", "Yes"], ["green", "red"]):
        subset = percentages[percentages["stroke"] == stroke]
        rects = ax.barh(
            subset[feature_name],
            subset["nr_of_patients"],
            height=0.5,
            label=stroke,
            color=color,
        )

        # Add data labels
        for rect in rects:
            width = rect.get_width()
            if stroke == "No":  # 'No' labels
                ax.text(
                    45,  # Fixed x position for 'No' labels
                    rect.get_y() + rect.get_height() / 2,
                    "{:.1f}%".format(width),
                    ha="left",
                    va="center",
                    color="white",
                    fontsize=12,
                )
            else:  # 'Yes' labels
                ax.text(
                    width + 2,  # Add a small value to move text to the right of the bar
                    rect.get_y() + rect.get_height() / 2,
                    "{:.1f}%".format(width),
                    ha="left",
                    va="center",
                    color="red",
                    fontsize=12,
                )
    ax.legend(title="Stroke", loc="upper right")
    ax.set_xlabel("Percentage of patients (%)")
    ax.set_title(f"Percentage of patients with stroke by {feature_name}")
    plt.show()


def show_distribution(data: pd.DataFrame, output: str, column: str) -> None:
    """Displays distribution of a specific column, split by output variable (stroke status).

    This function creates two side-by-side histograms of the specified column's data,
    separated by whether the patient had a stroke or not.

    Args:
        data (pd.DataFrame): The input data.
        output (str): The output or target variable, in this case 'stroke'.
        column (str): The specific column to display distribution for.
    """
    f, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.histplot(data[data[output] == "No"][column], ax=ax[0], kde=True, color="green")
    ax[0].set_title(f"{column} for people who did not have stroke")
    sns.histplot(data[data[output] == "Yes"][column], ax=ax[1], kde=True, color="red")
    ax[1].set_title(f"{column} for people who had stroke")
    plt.show()


def calculate_roc_auc(
    pipelines: List, X_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[List, List, pd.DataFrame]:
    """Calculates the ROC AUC score for each pipeline using cross-validation.

    This function performs cross-validation on each model in the pipeline,
    then stores and returns the mean and standard deviation of the ROC AUC scores.

    Args:
        pipelines (List): List of pipelines for each model.
        X_train (pd.DataFrame): The input features for training.
        y_train (pd.Series): The output variable for training.

    Returns:
        Tuple[List, List, pd.DataFrame]: Returns three outputs -
            1. List of model names.
            2. List of ROC AUC scores for each model.
            3. DataFrame comparing the mean and standard deviation of ROC AUC scores for each model.
    """
    model_name = []
    results_mean = []
    results_std = []
    roc_auc = []
    for pipe, model in pipelines:
        kfold = KFold(n_splits=5)
        crossv_results = cross_val_score(
            model, X_train, y_train, cv=kfold, scoring="roc_auc"
        )
        model_name.append(pipe[0:20])
        results_mean.append(crossv_results.mean())
        results_std.append(crossv_results.std())
        roc_auc.append(crossv_results)
    models_comparison = pd.DataFrame(
        {"CV mean": results_mean, "Std": results_std}, index=model_name
    )
    return model_name, roc_auc, models_comparison


def create_countplot(data: pd.DataFrame, list_of_columns: List[str]) -> None:
    """
    Creates countplots for categorical features of a dataset.

    Countplots offer an understanding of how many instances are represented by each specific discrete feature.
    In other words, it provides frequency counts of categorical features.

    Args:
        data (pd.DataFrame): The input dataset.
        list_of_columns (List[str]): The list of features to plot countplots for.

    Returns:
        None: The function returns None. It shows the plot using plt.show implicitly.
    """
    fig, axes = plt.subplots(1, len(list_of_columns), figsize=(26, 6))
    for i, ax in enumerate(axes.flatten()):
        sns.countplot(
            data=data,
            x=data[list_of_columns[i]],
            ax=ax,
            order=data[list_of_columns[i]].value_counts().index,
        )
        ax.set_xlabel("")
        ax.set_title(f"{' '.join(list_of_columns[i].split('_'))}", fontsize=13, y=1.03)
    sns.despine(left=True)
    plt.show()


def create_pie_charts(data: pd.DataFrame, list_of_columns: List[str]) -> None:
    """
    Creates pie charts for categorical features of a dataset.

    Pie charts depict the proportion of instances represented by each category of a discrete feature.

    Args:
        data (pd.DataFrame): Input dataframe.
        list_of_columns (List[str]): List of feature names for which pie charts will be created.

    Returns:
        None. Displays pie charts.

    """

    n_columns = len(list_of_columns)
    fig, axes = plt.subplots(1, n_columns, figsize=(5 * n_columns, 6))

    # If only one column is provided, axes will not be an array.
    # So, convert it to an array for consistency in the loop.
    if n_columns == 1:
        axes = np.array([axes])

    for ax, column in zip(axes.flatten(), list_of_columns):
        value_counts = data[column].value_counts()
        labels = value_counts.index
        values = value_counts.values

        ax.pie(values, labels=labels, autopct="%1.1f%%")
        ax.set_title(column.replace("_", " "), fontsize=13, y=1.03)

    plt.show()


def create_confusion_matrix(
    dict_of_models: Dict[str, any], X_test: np.ndarray, y_test: np.ndarray
) -> plt.figure:
    """
    Functions that create confusion matrix for machine learning model outcome.

    Args:
        dict_of_models (Dict[str, any]): Dictionary of models with names as keys and models as values.
        X_test (np.ndarray): numpy arrays with test data.
        y_test (np.ndarray): numpy arrays with test labels.

    Returns:
        confusion matrix (plt.figure).

    """

    fig, ax = plt.subplots(1, len(dict_of_models), figsize=(24, 4))

    for i, (key, value) in enumerate(dict_of_models.items()):
        y_pred = cross_val_predict(value, X_test, y_test, cv=6)
        sns.heatmap(confusion_matrix(y_test, y_pred), ax=ax[i], annot=True, fmt="2.0f")
        ax[i].set_title(f"Matrix for {key}")

    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4
    )
    plt.show()


#    return fig


def create_confusion_matrix_plots(
    dict_of_models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray
) -> plt.Figure:
    """
    Function that creates confusion matrix plots for machine learning model outcomes.

    Args:
        dict_of_models: Dictionary of models with names as keys and models as values.
        X_test: numpy array of test data.
        y_test: numpy array of test labels.

    Returns:
        Confusion matrix plots.
    """

    fig, ax = plt.subplots(1, len(dict_of_models), figsize=(24, 4))

    for i, (key, model) in enumerate(dict_of_models.items()):
        y_pred = cross_val_predict(model, X_test, y_test, cv=6)
        conf_mat = confusion_matrix(y_test, y_pred)
        sns.heatmap(conf_mat, ax=ax[i], annot=True, fmt="2.0f")
        ax[i].set_title(f"Matrix for {key}")

    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4
    )
    plt.show()


#    return fig


def print_confusion_matrix(
    dict_of_models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray
) -> None:
    """
    Function that prints confusion matrices for machine learning model outcomes.

    Args:
        dict_of_models: Dictionary of models with names as keys and models as values.
        X_test: numpy array of test data.
        y_test: numpy array of test labels.

    Returns:
        None
    """

    for key, model in dict_of_models.items():
        y_pred = cross_val_predict(model, X_test, y_test, cv=6)
        conf_mat = confusion_matrix(y_test, y_pred)

        # Print text representation of confusion matrix
        print(f"\nConfusion Matrix for {key}:")
        for j in range(conf_mat.shape[0]):
            row_str = " ".join(
                [f"{conf_mat[j, k]:>5}" for k in range(conf_mat.shape[1])]
            )
            print(row_str)

        col_names = [f"Pred {x}" for x in range(conf_mat.shape[1])]
        print(" ".join([f"{name:>5}" for name in col_names]))


def calculate_roc_auc_models(
    models: List, X_train: np.array, y_train: np.array, classifiers: List
) -> pd.DataFrame:
    """Calculate ROC AUC for multiple models and return a sorted dataframe with the results.

    Args:
        models (List): List of instantiated models to use for training and prediction.
        X_train (np.array): Training data.
        y_train (np.array): Labels for training data.
        classifiers (List): List of model names.

    Returns:
        sorted_df (pd.DataFrame): Dataframe with 'CV mean' and 'Std' as columns,
                                  and model names as index, sorted by 'CV mean' in descending order.
    """

    results = {}
    kfold = StratifiedKFold(n_splits=5)

    for name, model in zip(classifiers, models):
        cv_results = cross_val_score(
            model, X_train, y_train, cv=kfold, scoring="roc_auc"
        )
        results[name] = {"CV mean": cv_results.mean(), "Std": cv_results.std()}

    sorted_df = pd.DataFrame(results).T.sort_values(by=["CV mean"], ascending=False)

    return sorted_df


def create_confusion_matrix_for_list(
    models_list: List, X_test: np.array, y_test: np.array
) -> None:
    """Create confusion matrices for machine learning model outcomes.

    Args:
        models_list (List): List of trained models.
        X_test (np.array): Test dataset.
        y_test (np.array): Test labels.

    Returns:
        None. Displays confusion matrices as subplots.
    """
    n_models = len(models_list)
    f, ax = plt.subplots(1, n_models, figsize=(4 * n_models, 4))

    # If only one model is provided, axes will not be an array.
    # So, convert it to an array for consistency in the loop.
    if n_models == 1:
        ax = np.array([ax])

    for i, model in enumerate(models_list):
        y_pred = cross_val_predict(model, X_test, y_test, cv=6)
        sns.heatmap(confusion_matrix(y_test, y_pred), ax=ax[i], annot=True, fmt="2.0f")
        model_name = str(model).split("(")[0]  # Get the model name without arguments
        model_args = "\n".join(
            str(model).split("(")[1:]
        )  # Get the model arguments with line breaks
        model_args = model_args.rstrip(")")  # Remove the ending bracket
        ax[i].set_title(f"{model_name}\n{model_args}")

    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.8
    )
    plt.show()


def calculate_predictions(model, X_train, X_test, y_train):
    """Train the model using the training data and predict the labels of the test data.

    Args:
        model: The machine learning model to train and use for predictions.
        X_train (np.array): The training data.
        X_test (np.array): The test data.
        y_train (np.array): The labels of the training data.

    Returns:
        np.array: The predicted labels of the test data.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred


def create_heatmap(df: pd.DataFrame, size_of_figure: Tuple[int, int]) -> plt.figure:
    """Create a correlation heatmap from a dataframe.

    Args:
        df (pd.DataFrame): The dataframe containing the features to analyze.
        size_of_figure (Tuple[int, int]): The desired figure size.

    Returns:
        plt.figure: The correlation heatmap.
    """
    corr_data = df
    corr = corr_data.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))

    f, ax = plt.subplots(figsize=(size_of_figure))

    cmap = sns.color_palette("coolwarm", as_cmap=True)
    heatmap = sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        annot=True,
    )

    heatmap.set_title(
        f"Correlation heatmap of data attributes",
        fontdict={"fontsize": 16},
        pad=12,
    )
    plt.xlabel("")
    plt.ylabel("")
    plt.show()


#    return plt.gcf()


def plot_cat_boost(col, categorical_data):
    palette = ["green", "red"]

    data_group = categorical_data.groupby([col, "stroke"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(9, 3))
    data_group.plot(kind="barh", stacked=True, color=palette, ax=ax)

    # Add data labels
    for i, rect in enumerate(ax.patches):
        width = rect.get_width()
        if i < len(ax.patches) / 2:  # 'No' labels
            ax.text(
                rect.get_x() + 5,  # Add a small value to move text to the right
                rect.get_y() + rect.get_height() / 2,
                "%d" % int(width),
                ha="left",
                va="center",
                color="white",
                fontsize=12,
            )
        else:  # 'Yes' labels
            ax.text(
                rect.get_x()
                - 5,  # Subtract a small value to move text to the left of the bar
                rect.get_y() + rect.get_height() / 2,
                "%d" % int(width),
                ha="right",
                va="center",
                color="red",
                fontsize=12,
            )

    # Display value counts
    print(data_group)

    plt.title(f"Stroke Count in {col}")
    plt.xlabel("Counts")
    plt.ylabel(col)
    plt.show()
