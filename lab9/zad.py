from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def zad1():
    diabetes = pd.read_csv("lab9\diabetes.csv")
    num_cols = diabetes.columns.drop('Outcome')
    X = diabetes[num_cols]
    y = diabetes['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    tree = DecisionTreeClassifier(random_state = 2025)
    scores = cross_val_score(tree, X_train, y_train)

    params_grid = {
        'max_depth': range(4,9),
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

    y_pred = best_tree.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {acc} \nPrecision: {prec} \nRecall: {recall} \nF1: {f1}")


    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)


    plt.figure(figsize=(20,10))
    plot_tree(best_tree, feature_names=X.columns, filled=True,
              rounded=True)
    plt.show()

    ax = sns.heatmap(conf_matrix, annot=True, fmt='d', )
 
    # set x-axis label and ticks. 
    ax.set_xlabel("Predicted Diagnosis", fontsize=14, labelpad=20)
    ax.xaxis.set_ticklabels(['Negative', 'Positive'])
    
    # set y-axis label and ticks
    ax.set_ylabel("Actual Diagnosis", fontsize=14, labelpad=20)
    ax.yaxis.set_ticklabels(['Negative', 'Positive'])
    
    # set plot title
    ax.set_title("Confusion Matrix for the Diabetes Detection Model", fontsize=14, pad=20)
    
    plt.show()

def zad2():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    free_tree = DecisionTreeClassifier(random_state=2025)
    path = free_tree.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas

    a_trees = []

    for ccp_alpha in ccp_alphas:
        tree = DecisionTreeClassifier(random_state=2025, ccp_alpha=ccp_alpha)
        tree.fit(X_train, y_train)
        a_trees.append(tree)

    #dropping last tree, it is just root and disrupts graph
    a_trees = a_trees[:-1]
    ccp_alphas = ccp_alphas[:-1]

    train_scores = [tree.score(X_train, y_train) for tree in a_trees]
    test_scores = [tree.score(X_test, y_test) for tree in a_trees]

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    plt.show()


zad2()