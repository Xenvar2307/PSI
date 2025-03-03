import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

import dcor


def zad1():
    f = lambda x: (x)
    x_tr = np.linspace(0.0, 5, 200)
    y_tr = f(x_tr)
    x = stats.uniform(1, 3).rvs(100)
    y = f(x) + stats.norm(0, 0.1).rvs(len(x))

    plt.figure(figsize=(6, 6))
    axes = plt.gca()
    axes.set_xlim([0, 5])
    axes.set_ylim([0, 5])
    plt.plot(x_tr, y_tr, "--k")
    plt.plot(x, y, "ok", ms=3)
    plt.show()

    corr = {}
    corr["pearson"], _ = stats.pearsonr(x, y)
    corr["spearman"], _ = stats.spearmanr(x, y)
    corr["kendall"], _ = stats.kendalltau(x, y)
    corr["dcor"] = dcor.distance_correlation(x, y)
    print(corr)


def zad2():
    f = lambda x: (-x + 5)
    x_tr = np.linspace(0.0, 5, 200)
    y_tr = f(x_tr)
    x = stats.uniform(1, 3).rvs(100)
    y = f(x) + stats.norm(0, 0.1).rvs(len(x))

    plt.figure(figsize=(6, 6))
    axes = plt.gca()
    axes.set_xlim([0, 5])
    axes.set_ylim([0, 5])
    plt.plot(x_tr, y_tr, "--k")
    plt.plot(x, y, "ok", ms=3)
    plt.show()

    corr = {}
    corr["pearson"], _ = stats.pearsonr(x, y)
    corr["spearman"], _ = stats.spearmanr(x, y)
    corr["kendall"], _ = stats.kendalltau(x, y)
    corr["dcor"] = dcor.distance_correlation(x, y)
    print(corr)


def zad3():
    f = lambda x: (x)
    x_tr = np.linspace(0.0, 5, 200)
    y_tr = f(x_tr)
    x = stats.uniform(1, 3).rvs(100)
    y = f(x) + stats.norm(0, 0.5).rvs(len(x))

    plt.figure(figsize=(6, 6))
    axes = plt.gca()
    axes.set_xlim([0, 5])
    axes.set_ylim([0, 5])
    plt.plot(x_tr, y_tr, "--k")
    plt.plot(x, y, "ok", ms=3)
    plt.show()

    corr = {}
    corr["pearson"], _ = stats.pearsonr(x, y)
    corr["spearman"], _ = stats.spearmanr(x, y)
    corr["kendall"], _ = stats.kendalltau(x, y)
    corr["dcor"] = dcor.distance_correlation(x, y)
    print(corr)


def zad4():
    f = lambda x: x**2
    x_tr = np.linspace(-10, 10, 200)
    y_tr = f(x_tr)
    x = stats.uniform(-3, 6).rvs(100)
    y = f(x) + stats.norm(0, 0.3).rvs(len(x))

    plt.figure(figsize=(6, 6))
    axes = plt.gca()
    axes.set_xlim([-3, 3])
    axes.set_ylim([0, 10])
    # axes.set_ylim([np.min(y),f(5)])
    plt.plot(x_tr, y_tr, "--k")
    plt.plot(x, y, "ok", ms=3)
    plt.show()

    corr = {}
    corr["pearson"], _ = stats.pearsonr(x, y)
    corr["spearman"], _ = stats.spearmanr(x, y)
    corr["kendall"], _ = stats.kendalltau(x, y)
    corr["dcor"] = dcor.distance_correlation(x, y)
    print(corr)


def zad5():  # funkcja spoko, zakres trzeba lepiej dobrać a nie znam parametrów
    f = lambda x: 2**x
    x_tr = np.linspace(-10, 3, 200)
    y_tr = f(x_tr)
    x = stats.uniform(-10, 12).rvs(100)
    y = f(x) + stats.norm(0, 0.3).rvs(len(x))

    plt.figure(figsize=(6, 6))
    axes = plt.gca()
    axes.set_xlim([-10, 3])
    axes.set_ylim([0, 10])
    # axes.set_ylim([np.min(y),f(5)])
    plt.plot(x_tr, y_tr, "--k")
    plt.plot(x, y, "ok", ms=3)
    plt.show()

    corr = {}
    corr["pearson"], _ = stats.pearsonr(x, y)
    corr["spearman"], _ = stats.spearmanr(x, y)
    corr["kendall"], _ = stats.kendalltau(x, y)
    corr["dcor"] = dcor.distance_correlation(x, y)
    print(corr)


def zad6():  # jakaś niemonotoniczna
    return 0


zad5()
