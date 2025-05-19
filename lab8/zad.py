from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter

from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib.pyplot as plt

def zad1():
    cancer = load_breast_cancer()

    # get the data
    X = cancer.data
    y = cancer.target

    # split it into training and test sets
    X_train, X_test , y_train, y_test= train_test_split(X,y, random_state=5, test_size=.1)
    scaler = StandardScaler()


    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    fitted_model_sc = SVC().fit(X_train_scaled, y_train)
    fitted_model_no_sc = SVC().fit(X_train, y_train)

    y_pred_sc = fitted_model_sc.predict(X_test_scaled)
    y_pred_no_sc = fitted_model_no_sc.predict(X_test)

    print(y_pred_sc)
    print(y_pred_no_sc)
    print(y_test)
    print("accuracy with scaling: ", accuracy_score(y_test, y_pred_sc))
    print("accuracy without scaling: ", accuracy_score(y_test, y_pred_no_sc))

def zad2():
    # load dataset
    iris = load_iris()
    X, y = load_iris(return_X_y=True)
    # all of 0, 30% of 1 and 30% of 2
    X_cut = np.vstack((X[0:65,:], X[100:115,:]))
    y_cut = np.hstack((y[0:65], y[100:115]))

    k = 5

    kf = KFold(n_splits=k)
    strat_kf = StratifiedKFold(n_splits=k)

    model = LogisticRegression()


    print("NORMAL K FOLD")
    for i, (train, test) in enumerate(kf.split(X_cut, y_cut)):
        print("Fold:",i, end=" -> ")
        counter = Counter(y_cut[train])
        total = counter.total()
        print(f"Class share in -> Train: 0: {counter[0]/total:.2%} 1: {counter[1]/total:.2%} 2: {counter[2]/total:.2%}", end='  ')
        counter = Counter(y_cut[test])
        total = counter.total()
        print(f"Test: 0: {counter[0]/total:.2%} 1: {counter[1]/total:.2%} 2: {counter[2]/total:.2%}", end="  ")
        model.fit(X_cut[train], y_cut[train])
        y_pred = model.predict(X_cut[test])
        acc = accuracy_score(y_cut[test], y_pred)
        prec = precision_score(y_cut[test], y_pred, average='micro')
        recall = recall_score(y_cut[test], y_pred, average='micro')
        f1 = f1_score(y_cut[test], y_pred, average='micro')
        print(f"Metrics: acc={acc}, prec={prec}, recall={recall}, f1_score={f1}")

    print("Stratified K FOLD")
    for i, (train, test) in enumerate(strat_kf.split(X_cut, y_cut)):
        print("Fold:",i, end=" -> ")
        counter = Counter(y_cut[train])
        total = counter.total()
        print(f"Class share in -> Train: 0: {counter[0]/total:.2%} 1: {counter[1]/total:.2%} 2: {counter[2]/total:.2%}", end='  ')
        counter = Counter(y_cut[test])
        total = counter.total()
        print(f"Test: 0: {counter[0]/total:.2%} 1: {counter[1]/total:.2%} 2: {counter[2]/total:.2%}", end="  ")
        model.fit(X_cut[train], y_cut[train])
        y_pred = model.predict(X_cut[test])
        acc = accuracy_score(y_cut[test], y_pred)
        prec = precision_score(y_cut[test], y_pred, average='micro')
        recall = recall_score(y_cut[test], y_pred, average='micro')
        f1 = f1_score(y_cut[test], y_pred, average='micro')
        print(f"Metrics: acc={acc}, prec={prec}, recall={recall}, f1_score={f1}")

def zad3():
     # data preperation
    iris = load_iris()
    X, y = load_iris(return_X_y=True)
    # 
    y[:] = [1 if x!=0 else 0 for x in y]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2025)

    params_grid = {
        'model__C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
        'model__penalty': ['l1', 'l2'] # tylko takie wspiera solver liblinear
    }

    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(random_state=2025, solver="liblinear"))
    ])

    search = GridSearchCV(pipeline, params_grid)
    search.fit(X_train, y_train)
    y_pred = search.predict(X_test)
    print("Accuracy: ",accuracy_score(y_test, y_pred))

    fpr, tpr, thresholds = roc_curve(y_test, y_pred) 
    roc_auc = roc_auc_score(y_test, y_pred) 

    plt.figure()  
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

zad1()