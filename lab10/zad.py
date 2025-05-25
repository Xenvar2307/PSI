# Importy niezbędnych paczek
import numpy as np
from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from scipy.stats import uniform


def zad1():
    X, y = make_classification()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    logReg = LogisticRegression(random_state=2025)
    tree = DecisionTreeClassifier(random_state=2025)
    svm = SVC(random_state=2025, probability=True)

    logReg.fit(X_train, y_train)
    logReg_pred_proba = logReg.predict_proba(X_test)

    tree.fit(X_train, y_train)
    tree_pred_proba = logReg.predict_proba(X_test)

    svm.fit(X_train, y_train)
    svm_pred_proba = svm.predict_proba(X_test)

    print("True data for y_test:")
    print(y_test)

    mean_proba = (logReg_pred_proba + tree_pred_proba + svm_pred_proba)/3
    #print(mean_proba)
    soft_voting_classes = np.argmax(mean_proba, axis=1)
    print("Manual Soft Voting logReg, tree, svm: ")
    print(soft_voting_classes, f" accur: {accuracy_score(y_test, soft_voting_classes)}")

    clf1 = DecisionTreeClassifier(random_state=2025)
    clf2 = LogisticRegression(random_state=2025)
    clf3 = SVC(random_state=2025, probability=True)
    autoVotingClassifier = VotingClassifier(estimators=[('dec_tree', clf1),
                                                         ('lin_reg', clf2),
                                                         ('svm', clf3)]
                                            ,voting="soft")
    autoVotingClassifier.fit(X_train, y_train)
    auto_voting_classes = autoVotingClassifier.predict(X_test)

    print("Auto Soft Voting VotingClassifier: ")
    print(auto_voting_classes, f" accur: {accuracy_score(y_test, auto_voting_classes)}")

def zad2():
    X, y = make_classification()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=50, random_state=2025)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("True data for y_test:")
    print(y_test)
    print("ForestClassifier with 50 trees: ")
    print(y_pred, f" accur: {accuracy_score(y_test, y_pred)}")

def zad3():
    X, y = make_moons(n_samples=500, noise = 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = XGBClassifier(n_estimators=100, learning_rate = 0.1, random_state=2025)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("True data for y_test:")
    print(y_test)
    print("XGBClassifier with 50 trees: ")
    print(y_pred, f" accur: {accuracy_score(y_test, y_pred)}")
    
def zad4():
    X, y = make_moons(n_samples=500, noise = 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    xgb = XGBClassifier(random_state=2025)
    distribution = dict(
        n_estimators=range(10,201, 10),
        learning_rate = uniform(loc= 0.01, scale= 0.5),
        max_depth = range(3,12,1),
        min_child_weight = uniform(loc= 0.5, scale = 1.0),
        subsample = uniform(loc= 0.5, scale=0.5)
    )
    classifier = RandomizedSearchCV(xgb,param_distributions= distribution , random_state=2025)
    search = classifier.fit(X_train, y_train)

    print(search.best_score_)
    print(search.best_params_)

zad4()