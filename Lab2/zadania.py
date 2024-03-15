import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import normalize

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
def zad3():
    A = np.random.randint(5, 15 + 1, 100)
    print(np.bincount(A))
    print(np.argmax(np.bincount(A)))


def zad4():
    X, y = load_breast_cancer(return_X_y=True)
    # first part of the task
    y[y == 0] = -1

    # second part of the task

    maks = np.max(X, axis=0)
    min = np.min(X, axis=0)

    print(X)

    X = (X - min) / (maks - min)

    print(X)


# PANDAS


def zad5():
    df = pd.read_csv("Lab2/data/airports.csv")
    print(pd.unique(df["iso_country"]))
    print("\n\n")
    print(df["name"].tail(12))

    print("\n")

    # chodzi o to że może się różnić indeks 1 i wartość id 1
    print(df.loc[1])
    print(df.iloc[1])

    print("\n")

    only_country = df.loc[df.iso_country == "PR", ["name", "iso_country"]]
    only_country = only_country.sort_values("name")
    only_country = only_country.drop_duplicates()

    print(only_country)

    print("\n\n")
    print(df[df.name != df.municipality][["name", "municipality"]])
    print("\n\n")
    print(df[["elevation_ft"]])
    df["elevation_ft"] = df["elevation_ft"].fillna(0)
    df["elevation_ft"] = df["elevation_ft"].apply(lambda x: x * 0.3048)
    df.rename(columns={"elevation_ft": "elevation_m"}, inplace=True)
    print(df["elevation_m"])

    print("\n\n")
    no_airports_per_country = df["iso_country"].value_counts()
    countries_1_airport = no_airports_per_country[no_airports_per_country == 1].index
    print(countries_1_airport)
    # print(df.loc[df.iso_country.isin(countries_1_airport)])


def zad6():
    df = pd.read_csv("https://github.com/Ulvi-Movs/titanic/raw/main/train.csv")
    print(df.shape)
    df = df.drop(columns=["PassengerId", "Name", "Ticket"])
    print(df.head())
    df.loc[df.Cabin.isnull()]
    # checking for nan
    df["Cabin"] = df["Cabin"].apply(lambda x: "Nie" if x != x else "Tak")
    print(df.head())


def zad7():
    df = pd.read_csv("https://github.com/Ulvi-Movs/titanic/raw/main/train.csv")
    print(df.shape)
    df = df.drop(columns=["PassengerId", "Name", "Ticket"])
    print(df.head())
    df.loc[df.Cabin.isnull()]
    # checking for nan
    df["Cabin"] = df["Cabin"].apply(lambda x: "Nie" if x != x else "Tak")
    df["HasCabin"] = df["Cabin"].apply(lambda x: 1 if x == "Tak" else 0)
    print(df.head())


def zad8():
    df = pd.read_csv("https://github.com/Ulvi-Movs/titanic/raw/main/train.csv")
    print(df.dropna(axis=1))
    print(df.dropna(axis=0))


zad8()
