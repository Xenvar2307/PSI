from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import make_moons
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt


def zad1():
    # data preperation
    iris = datasets.load_iris()
    X, y = datasets.load_iris(return_X_y=True)
    # only petal width
    X = X[:, 3].reshape(-1,1)
    # 1 Virginica 0 something else
    y[:] = [1 if x==2 else 0 for x in y]
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2025)

    # find best C
    model = LogisticRegression(max_iter=200)
    #print(model.get_params())
    C_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000]
    C_values_accuracy = [0]*len(C_values)

    param_space = {
        'C': C_values
    }

    grid_search = GridSearchCV(model, param_space, scoring='accuracy', cv = 5)
    fitted_model = grid_search.fit(X_train, y_train)
    #print(grid_search.cv_results_)
    print(fitted_model.best_params_)

    #check all C values on test set for plot
    for i in range(10):
        model = LogisticRegression(max_iter=200, C=C_values[i])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        #print(y_pred)
        #print(y_test)
        #print(np.mean(y_pred == y_test))
        C_values_accuracy[i] = np.mean(y_pred == y_test)

    plt.subplot(1,2,1)
    plt.title('accuracy based on C value')
    plt.plot(C_values, C_values_accuracy)
    plt.xscale('log')
    plt.xlabel('C value')
    plt.ylabel('accuracy')

    plt.subplot(1,2,2)
    plt.title('Probability of being Virginica with decision boundary')
    scatter_plot = plt.scatter(X_test, fitted_model.predict_proba(X_test)[:,1], c= y_test)
    coef = fitted_model.best_estimator_.coef_[0,0]
    intercept = fitted_model.best_estimator_.intercept_[0]
    print (coef, intercept)
    boundary = (0.5 - intercept)/coef
    plt.plot([boundary, boundary], [0,1], 'orange', linewidth = 6, label= 'Boundary')
    plt.xlabel('Petal width')
    plt.ylabel('Chance of being Virginica')

    plt.legend(handles=scatter_plot.legend_elements()[0], labels=['Not Virginica', 'Virginica'])
    plt.show()

def zad2():
     # data preperation
    iris = datasets.load_iris()
    X, y = datasets.load_iris(return_X_y=True)
    # only petal width
    X = X[:, 2:4]#.reshape(-1,2)
    # 1 Virginica 0 something else
    y[:] = [1 if x==2 else 0 for x in y]
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2025)

    # find best C
    model = LogisticRegression(max_iter=200)
    #print(model.get_params())
    C_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000]
    C_values_accuracy = [0]*len(C_values)

    param_space = {
        'C': C_values
    }

    grid_search = GridSearchCV(model, param_space, scoring='accuracy', cv = 5)
    fitted_model = grid_search.fit(X_train, y_train)
    #print(grid_search.cv_results_)
    print(fitted_model.best_params_)

    #check all C values on test set for plot
    for i in range(10):
        model = LogisticRegression(max_iter=200, C=C_values[i])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        #print(y_pred)
        #print(y_test)
        #print(np.mean(y_pred == y_test))
        C_values_accuracy[i] = np.mean(y_pred == y_test)

    plt.subplot(1,2,1)
    plt.title('accuracy based on C value')
    plt.plot(C_values, C_values_accuracy)
    plt.xscale('log')
    plt.xlabel('C value')
    plt.ylabel('accuracy')


    plt.subplot(1,2,2)
    plt.title('Estimator boundary')

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    Z = fitted_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha = 0.8)

    scatter_plot = plt.scatter(X_test[:, 0],X_test[:,1], c= y_test, edgecolors='k')

    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    

    plt.legend(handles=scatter_plot.legend_elements()[0], labels=['Not Virginica', 'Virginica'])
    plt.show()

def zad3ver1():
    moons_data = make_moons(n_samples=300, noise=0.25, random_state=42)
    X = moons_data[0]
    y = moons_data[1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2025)

    model = SVC()

    param_space = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }

    grid_search = GridSearchCV(model, param_space, scoring='f1', cv = 5)
    fitted_model = grid_search.fit(X_train, y_train)
    print("Best kernel for F1 score: ",fitted_model.best_params_)

    grid_search = GridSearchCV(model, param_space, scoring='accuracy', cv = 5)
    fitted_model = grid_search.fit(X_train, y_train)
    print("Best kernel for accuracy: ",fitted_model.best_params_)

    grid_search = GridSearchCV(model, param_space, scoring='precision', cv = 5)
    fitted_model = grid_search.fit(X_train, y_train)
    print("Best kernel for precision: ",fitted_model.best_params_)

    model = SVC(kernel='rbf')
    fitted_model = model.fit(X_train,y_train)

    plt.title('Estimator boundary')

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    Z = fitted_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha = 0.8)

    scatter_plot = plt.scatter(X_test[:, 0],X_test[:,1], c= y_test, edgecolors='k')

    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.xlabel('X')
    plt.ylabel('Y')
    

    plt.legend(handles=scatter_plot.legend_elements()[0], labels=['0', '1'])
    plt.show()

def zad3ver2():
    # treating other values as hiperparameters???
    # late night thought, maybe too late night to be a thought
    moons_data = make_moons(n_samples=300, noise=0.25, random_state=42)
    X = moons_data[0]
    y = moons_data[1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2025)

    model = SVC()

    param_space = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  

    grid_search = GridSearchCV(model, param_space, scoring='f1', cv = 5)
    fitted_model = grid_search.fit(X_train, y_train)
    print("Best params for F1 score: ",fitted_model.best_params_)

    grid_search = GridSearchCV(model, param_space, scoring='accuracy', cv = 5)
    fitted_model = grid_search.fit(X_train, y_train)
    print("Best params for accuracy: ",fitted_model.best_params_)

    #grid_search = GridSearchCV(model, param_space, scoring='precision', cv = 5)
    #fitted_model = grid_search.fit(X_train, y_train)
    #print("Best params for precision: ",fitted_model.best_params_)

    model = SVC(kernel='rbf')
    fitted_model = model.fit(X_train,y_train)

    plt.title('Estimator boundary')

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    Z = fitted_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha = 0.8)

    scatter_plot = plt.scatter(X_test[:, 0],X_test[:,1], c= y_test, edgecolors='k')

    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.xlabel('X')
    plt.ylabel('Y')
    

    plt.legend(handles=scatter_plot.legend_elements()[0], labels=['0', '1'])
    plt.show()

zad3ver2()