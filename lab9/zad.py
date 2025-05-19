from sklearn.datasets import load_diabetes
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
import pandas as pd
import matplotlib.pyplot as plt



def zad1():
    diabetes = pd.read_csv("lab9\diabetes.csv")
    num_cols = diabetes.columns.drop('Outcome')
    X = diabetes[num_cols]
    y = diabetes['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    tree = DecisionTreeClassifier(random_state = 2025)
    scores = cross_val_score(tree, X_train, y_train)

    params_grid = {
        'max_depth': range(4,11),
        'min_samples_split': range(10,111,10),
        'criterion': ['gini','entropy']
    }

    cross_validation = StratifiedKFold(n_splits=10)

    search = GridSearchCV(tree, param_grid=params_grid, cv = cross_validation)
    search.fit(X_train, y_train)
    print("Best Params:\n",search.best_params_,'\n')
    #print(search.cv_results_)

    best_tree = DecisionTreeClassifier(criterion=search.best_params_['criterion'],
                                    max_depth=search.best_params_['max_depth'],
                                    min_samples_split=search.best_params_['min_samples_split'], random_state = 2025)
    best_tree.fit(X_train, y_train)
    result = dict(zip(X.keys(), best_tree.feature_importances_))
    result.update((key, float(value)) for key, value in result.items())

    sorted_result = sorted(
        result.items(),
        key = lambda kv: kv[1],
        reverse=True
    )
    print("Importance of features: \n",sorted_result)

    plt.figure(figsize=(20,10))
    plot_tree(best_tree, feature_names=X.columns)
    plt.show()


zad1()