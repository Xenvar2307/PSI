import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sklearn.metrics as sklmetrics

def zad0():
    f = lambda x: ((x*2-1)*(x**2-2)*(x-2)+3)
    x_tr = np.linspace(0, 3, 200)
    y_tr = f(x_tr)
    x = stats.uniform(0,3).rvs(100)
    y = f(x) + stats.norm(0,0.2).rvs(len(x))
    M1 = np.vstack( (np.ones_like(x), x) ).T
    p1 = np.linalg.lstsq(M1, y, rcond=None)
    f_lr_1 = lambda x: p1[0][1] * x +p1[0][0]
    x_f_lr = np.linspace(0., 3, 200)
    y_f_lr = f_lr_1(x_tr)
    plt.figure(figsize=(6,6))
    axes = plt.gca()
    axes.set_xlim([0,3])
    axes.set_ylim([0,8])
    plt.plot(x_tr[:200], y_tr[:200], '--k')
    plt.plot(x_f_lr, y_f_lr, 'g')
    plt.plot(x, y, 'ok', ms=10)
    plt.show()

def zad1():
    f = lambda x: ((x*2-1)*(x**2-2)*(x-2)+3)
    x_tr = np.linspace(0, 3, 200)
    y_tr = f(x_tr)
    x = stats.uniform(0,3).rvs(100)
    y = f(x) + stats.norm(0,0.2).rvs(len(x))

    M1 = np.vstack( (np.ones_like(x), x, np.square(x)) ).T
    p1 = np.linalg.lstsq(M1, y, rcond=None)
    f_lr_1 = lambda x: p1[0][2] * (x**2) + p1[0][1] * x +p1[0][0]
    x_f_lr = np.linspace(0., 3, 200)
    y_f_lr = f_lr_1(x_tr)

    plt.figure(figsize=(6,6))
    axes = plt.gca()
    axes.set_xlim([0,3])
    axes.set_ylim([0,8])
    plt.plot(x_tr[:200], y_tr[:200], '--k')
    plt.plot(x_f_lr, y_f_lr, 'g')
    plt.plot(x, y, 'ok', ms=10)
    plt.show()

def zad2():
    f = lambda x: ((x*2-1)*(x**2-2)*(x-2)+3)
    x_tr = np.linspace(0, 3, 200)
    y_tr = f(x_tr)
    x = stats.uniform(0,3).rvs(100)
    y = f(x) + stats.norm(0,0.2).rvs(len(x))

    M1 = np.vstack( (np.ones_like(x), x, np.power(x,2),np.power(x,3)) ).T
    p1 = np.linalg.lstsq(M1, y, rcond=None)
    f_lr_1 = lambda x: p1[0][3] * (x**3) + p1[0][2] * (x**2) + p1[0][1] * x +p1[0][0]
    x_f_lr = np.linspace(0., 3, 200)
    y_f_lr = f_lr_1(x_tr)

    plt.figure(figsize=(6,6))
    axes = plt.gca()
    axes.set_xlim([0,3])
    axes.set_ylim([0,8])
    plt.plot(x_tr[:200], y_tr[:200], '--k')
    plt.plot(x_f_lr, y_f_lr, 'g')
    plt.plot(x, y, 'ok', ms=10)
    plt.show()

def zad3():
    f = lambda x: ((x*2-1)*(x**2-2)*(x-2)+3)
    x_tr = np.linspace(0, 3, 200)
    y_tr = f(x_tr)
    x = stats.uniform(0,3).rvs(100)
    y = f(x) + stats.norm(0,0.2).rvs(len(x))

    M1 = np.vstack( (np.ones_like(x), x, np.square(x)) ).T
    p1 = np.linalg.lstsq(M1, y, rcond=None)
    f_lr_1 = lambda x: p1[0][2] * (x**2) + p1[0][1] * x +p1[0][0]
    x_f_lr_1 = np.linspace(0., 3, 200)
    y_f_lr_1 = f_lr_1(x_tr)

    M2 = np.vstack( (np.ones_like(x), x, np.power(x,2),np.power(x,3)) ).T
    p2 = np.linalg.lstsq(M2, y, rcond=None)
    f_lr_2 = lambda x: p2[0][3] * (x**3) + p2[0][2] * (x**2) + p2[0][1] * x +p2[0][0]
    x_f_lr_2 = np.linspace(0., 3, 200)
    y_f_lr_2 = f_lr_2(x_tr)

    #use sklearn.metrics dla R**2 na przykład
    print ("Square function R2 score: ",sklmetrics.r2_score(y,f_lr_1(x)))

    print("Triangle function R2 score: ",sklmetrics.r2_score(y,f_lr_2(x)))

def zad4():  
    # Ładujemy dataset
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    boston = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([boston.values[::2, :], boston.values[1::2, :2]])
    target = boston.values[1::2, 2]

    # Preprocessing
    bos=pd.DataFrame(boston)
    bos=pd.DataFrame(data)
    feature_name = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO','B', 'LSTAT']
    bos.columns = feature_name
    bos['PRICE']=target # To jest nasza zmienna zależna
    bos.head()

    mod = smf.ols(formula='PRICE ~ CRIM + ZN + CHAS + NOX + RM + DIS + RAD + B + PTRATIO + LSTAT', data=bos)
    # nie zawarłem INDUS oraz AGE, bo miał wysokie P>|t| czyli szansę na niski wkład, a ilość zmiennych jest bardzo duża już -- R2: 0.741
    # próba usunięcia zmiennych o małej zależności: TAX czy B, R2 spada; 

    result = mod.fit()

    print(result.summary())

def zad5():
    df_adv = pd.read_csv('https://raw.githubusercontent.com/przem85/bootcamp/master/statistics/Advertising.csv', index_col=0)
    df_adv.head()

    mod = smf.ols(formula='sales ~ TV + radio', data= df_adv)
    # newspaper ma wysokie P>|t| wskazujące na niski wpływ, a mamy użyc tylko dwóch predykatorów
    # po usunięciu newspaper nie zmienia się znacząco miara R2
    result = mod.fit()

    print(result.summary())

zad5()