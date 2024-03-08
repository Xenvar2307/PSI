import numpy as np
import pandas as pd

X = np.array([[1, 2, 3], [4, 5, 6]])
# ones, zeros i wymiary daje wypelniona jedynkami i zerami
# eye, macierz jednostkowa
# arange z krokiem
# linspace, podzielenie na rowne przedzialy
# reshape, zmiana wymiarow


def zad1():
    points = np.random.rand(100, 10)
    for x in points:
        for y in points:
            d_square = 0
            for i in range(len(x)):
                d_square += (x[i] - y[i]) ** 2
            print(np.sqrt(d_square))


def zad2():
    mean = np.zeros(5)  # ?
    cov = np.eye(5)
    points = np.random.multivariate_normal(mean, cov, 100)
    means = np.mean(points, axis=0)
    standard = np.std(points, axis=0)
    wynik = (points - means) / standard  # temp
    print(np.cov(wynik, rowvar=False))
    print(np.mean(wynik))


# zad 1-4 numpy, pojebane tempo, nie zdążyłem nawet przeczytać


def zad4_5():
    df = pd.read_csv("Lab2/data/airports.csv")
    print(df["name"].tail(12))

    print("\n")

    print(df.loc[1])
    print(df.iloc[1])
    print(df.loc[1] == df.iloc[1])

    print("\n")

    print(df[df.iso_country == "US"].sort_values("name"))

    print(df[df.name != df.municipality])
    # do dokończenia


zad4_5()
