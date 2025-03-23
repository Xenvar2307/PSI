import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
import scipy.stats as stats
import pandas as pd
from scipy.stats import multivariate_normal

def zad1_2_3():
    #part1
    mean = [0,0]
    cov = [[4.40, -2.75],[-2.75, 5.50]]
    points = np.random.multivariate_normal(mean, cov, 1000)

    plt.scatter(points[:,0], points[:,1])
    #plt.show()

    #part2
    print(np.mean(points, axis = 0))
    print(np.cov(points,rowvar=False))

    #part3
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    print("Eigenvalues: ", eigenvalues)
    print("Eigenvectors:\n ", eigenvectors)

    plt.scatter(eigenvectors[:,0], eigenvectors[:,1], label="Eigenvectors")
    plt.legend(loc = 'upper right')

    xx, yy = np.meshgrid(np.linspace(-10,10,100),np.linspace(-10,10,100))
    xy = np.column_stack([xx.flat, yy.flat])
    zz = multivariate_normal.pdf(xy, mean, cov).reshape(xx.shape)

    plt.contour(xx,yy,zz, levels=6)

    plt.show()

def zad4():
    f = lambda x: (x)
    x_tr = np.linspace(0., 5, 200)
    y_tr = f(x_tr)
    x = stats.uniform(1,3).rvs(100)
    y = f(x) + stats.norm(0,0.1).rvs(len(x))

    plt.figure(figsize=(6,6));
    axes = plt.gca()
    axes.set_xlim([0,5])
    axes.set_ylim([0,5])
    plt.plot(x_tr, y_tr, '--k');
    plt.plot(x, y, 'ok', ms=3);
    plt.show()

    corr = {}
    corr["pearson"], _ = stats.pearsonr(x, y)
    corr["spearman"], _ = stats.spearmanr(x, y)
    corr["kendall"], _ = stats.kendalltau(x, y)
    print(corr)

def zad5():
    f = lambda x: (x)
    x_tr = np.linspace(0., 5, 200)
    y_tr = f(x_tr)
    x = stats.uniform(1,3).rvs(100)
    y = f(x) + stats.norm(0,0.5).rvs(len(x))

    plt.figure(figsize=(6,6));
    axes = plt.gca()
    axes.set_xlim([0,5])
    axes.set_ylim([0,5])
    plt.plot(x_tr, y_tr, '--k');
    plt.plot(x, y, 'ok', ms=3);
    plt.show()

    corr = {}
    corr["pearson"], _ = stats.pearsonr(x, y)
    corr["spearman"], _ = stats.spearmanr(x, y)
    corr["kendall"], _ = stats.kendalltau(x, y)
    print(corr)

def zad6():
    f = lambda x: 2**x
    x_tr = np.linspace(0., 30, 400)
    y_tr = f(x_tr)
    x = stats.uniform(0,30).rvs(300)
    y = f(x) + stats.norm(0,0.5).rvs(len(x))

    plt.figure(figsize=(6,6));
    axes = plt.gca()
    axes.set_xlim([0,30])
    axes.set_ylim([0,30])
    plt.plot(x_tr, y_tr, '--k');
    plt.plot(x, y, 'ok', ms=3);
    plt.show()

    corr = {}
    corr["pearson"], _ = stats.pearsonr(x, y)
    corr["spearman"], _ = stats.spearmanr(x, y)
    corr["kendall"], _ = stats.kendalltau(x, y)
    print(corr)

def zad7():
    points = np.random.rand(100,10)
    points[:,:1] = points[:,2:3] * 2
    points[:,4:5] = points[:,2:3] * 4
    points[:,6:7] = points[:,7:8] * (-4)


    df = pd.DataFrame(points)
    print (df)
    print (df.corr())
    

zad7()