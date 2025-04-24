import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.linear_model import Lasso
from sklearn.datasets import fetch_openml

def zad1():
    # 1. Generowanie danych
    np.random.seed(42)
    X = np.linspace(0, 1, 100)
    y = np.sin(2 * np.pi * X) + np.random.normal(0, 0.2, X.shape)
    X = X[:, np.newaxis]

    # 2. Podział danych (60% trening, 20% walidacja, 20% test)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    max_degree = 15
    MSE_val_list = list()
    MSE_test_list = list()
    MSE_x = range(1,max_degree+1)
    best_model = None
    best_MSE = None
    best_degree = None

    # trening każdego stopnia
    for degree in range(1,max_degree+1):
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X_train, y_train)

        # needed for second point of the task
        y_pred_val = model.predict(X_val)
        MSE_val = mean_squared_error(y_val, y_pred_val)
        MSE_val_list.append(MSE_val)

        # used later for plot
        y_pred_test = model.predict(X_test)
        MSE_test = mean_squared_error(y_test, y_pred_test)
        MSE_test_list.append(MSE_test)

        if ((best_MSE is None ) or best_MSE > MSE_val):
            best_MSE = MSE_val
            best_model = model
            best_degree = degree

    #plot MSE
    plt.subplot(1,2,1)
    plt.title("MSE per set")
    plt.plot(MSE_x, MSE_val_list, label = "validation set")
    plt.plot(MSE_x, MSE_test_list, label = "test set")
    plt.scatter([best_degree],[best_MSE], color = "red", label=f"best model: {best_degree}")

    plt.legend()
    plt.grid(True)

    #plot best model
    plt.subplot(1,2,2)
    plt.title("Best model prediction")
    plt.scatter(X_test, y_test, color='red', label='Dane treningowe')
    plt.plot(X, np.sin(2 * np.pi * X), label='Funkcja prawdziwa', color='green', linestyle='dashed')
    plt.plot(X, best_model.predict(X), label=f'Model (stopień {best_degree})', color='blue')

    plt.legend()
    plt.grid(True)

    plt.show()

def zad2():
    # walidacja krzyżowa
    df_adv = pd.read_csv('https://raw.githubusercontent.com/przem85/bootcamp/master/statistics/Advertising.csv', index_col=0)
    X = df_adv[['TV', 'radio','newspaper']]
    y = df_adv['sales']

    # dla 15 wykres nie miał sensu przez skalę, do 4 najlepiej widać najlepszy wynik
    max_degree = 15
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
   
    model = make_pipeline(PolynomialFeatures(), LinearRegression())
    
    #print(model.get_params())
    param_space = {
        'polynomialfeatures__degree': range(1,max_degree+1)
    }

    #grid search with basic metric R2
    grid_search_R2 = GridSearchCV(model, param_space, cv = 5)# jak sie nie poda cv to domyslnie jest 5 fold
    grid_search_R2.fit(X_train, y_train)
    print(f"Best degree: {grid_search_R2.best_params_}")

    def MSE_scorer():
        return make_scorer(mean_squared_error, greater_is_better=False)

    grid_search_MSE = GridSearchCV(model, param_space, scoring=MSE_scorer(), cv = 5)# jak sie nie poda cv to domyslnie jest 5 fold
    grid_search_MSE.fit(X_train, y_train) 
    MSE_scores = grid_search_MSE.cv_results_['mean_test_score'] * -1
    print(MSE_scores)

    # scores are printed as negatives, because of the comparison purposes
    # zamiast zamieniać większe = lepsze to wewnątrz klasy obracają score na ujemny i guess
    plt.title("mean score of folds (MSE) per degree")
    plt.plot(range(1, max_degree+1), MSE_scores)
    plt.show()

def zad3():
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    boston = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([boston.values[::2, :], boston.values[1::2, :2]])
    target = boston.values[1::2, 2]

    bos=pd.DataFrame(boston)
    bos=pd.DataFrame(data)
    feature_name = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO','B', 'LSTAT']
    bos.columns = feature_name
    bos['PRICE']=target # To jest nasza zmienna zależna

    X = bos[feature_name]
    y = bos['PRICE']

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    model = Ridge()
    #print(model.get_params())
    param_space = {
        'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    }
    grid_search = GridSearchCV(model, param_space, cv = 5)# jak sie nie poda cv to domyslnie jest 5 fold
    fitted_model = grid_search.fit(X_train, y_train)

    print(f"Best alpha: {grid_search.best_params_}")
    print(mean_squared_error(y_test, fitted_model.predict(X_test)))

def zad4():
    #data = fetch_openml("energy")
    # nie ma takiego datasetu, najblizsze co znalazlem to energy_efficiency
    # wiec po prostu zrobie na tym datasetcie, bo nie ma co sie doszukiwac bledu
    # najwyzej wyniki beda dziwne
    data = fetch_openml("energy_efficiency", version=2)
    X, y = fetch_openml("energy_efficiency", version=2, return_X_y=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    model = Lasso()
    #print(model.get_params())
    param_space = {
        'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    }
    grid_search = GridSearchCV(model, param_space, cv = 5)# jak sie nie poda cv to domyslnie jest 5 fold
    fitted_model = grid_search.fit(X_train, y_train)

    print(f"Best alpha: {grid_search.best_params_}")
    print(mean_squared_error(y_test, fitted_model.predict(X_test)))


zad4()