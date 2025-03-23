import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_breast_cancer

# zad 1: np dot new axes or reshape
# zad 9:min - sigma min + sigma, whatever that means
def isNaN(num):
       return num != num

#NumPy
def zad1():
        points = np.random.rand(100, 10)
        distances = np.linalg.norm(points - points.reshape(100,1,10), axis = 2)
        print(distances)

def zad2():
        mean = np.zeros(5)
        cov = np.eye(5)
        points = np.random.multivariate_normal(mean, cov, 100)

        wynik = (points - points.mean(axis = 0)) / points.std(axis=0)  # temp
        print(np.cov(wynik, rowvar=False))
        print(np.mean(wynik))

def zad3():
        points = np.random.randint(11,size= 100) + 5 
        print(np.bincount(points))
        print(np.argmax(np.bincount(points)))

def zad4():
        X, y  = load_breast_cancer(return_X_y=True)
        print(X.shape)
        print(y.shape)

        print(y)
        y[y == 0] = -1
        print(y)
        
        X_scaled = (X - X.min(axis=0)) / (X.max(axis=0)-X.min(axis=0))
        print(X)
        print(X_scaled)

        print(X_scaled.min(axis=0))
        print(X_scaled.max(axis=0))

#Pandas
def zad5():
    df = pd.read_csv("lab2/airports.csv")
    #print(pd.unique(df["iso_country"]))
    #print("\n\n")
    print(df["iso_country"].tail(12))

    print("\n")

    # chodzi o to że może się różnić indeks 1 i wartość id 1
    print("LOC[1]:\n",df.loc[1])
    print("ILOC[1]:\n",df.iloc[1])

    print("\n")

#wszystkie lotniska gdzies bo nie znalazlem kodu polski
    only_country = df.loc[df.iso_country == "PR", ["name", "iso_country"]]
    only_country = only_country.sort_values("name")
    only_country = only_country.drop_duplicates()
    print(only_country)

    print("\n\n")

#inne nazwy
    print(df[df.name != df.municipality][["name", "municipality"]])
    print("\n\n")

#elevation
    print(df[["elevation_ft"]])
    #df["elevation_ft"] = df["elevation_ft"].fillna(0)
    df["elevation_ft"] = df["elevation_ft"].apply(lambda x: float('nan') if isNaN(x) else x * 0.3048)
    df.rename(columns={"elevation_ft": "elevation_m"}, inplace=True)
    print(df["elevation_m"])
    print("\n\n")

#1 lotnisko
    no_airports_per_country = df["iso_country"].value_counts()
    countries_1_airport = no_airports_per_country[no_airports_per_country == 1].index
    print(countries_1_airport)
    # print(df.loc[df.iso_country.isin(countries_1_airport)])

def zad6_7():
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
    print(df.dropna())
    #print(df.dropna(axis=0))

def zad9a():
        inFile = 'http://ww2.amstat.org/publications/jse/datasets/babyboom.dat.txt'
        data = pd.read_csv(inFile, sep='[ ]+', header=None, engine='python',names= ['sex', 'Weight', 'Minutes'])
        df = data[['Minutes', 'sex', 'Weight']]
        
        x = df.Weight[df['sex'] == 1].values
        y = df.Weight[df['sex'] == 2].values

        plt.scatter(np.arange(len(x)),x, label = 'sex1')
        plt.scatter(np.arange(len(y)),y,label = 'sex2')
        plt.legend(loc = 'upper right')
        plt.show()

def zad9b():
        inFile = 'http://ww2.amstat.org/publications/jse/datasets/babyboom.dat.txt'
        data = pd.read_csv(inFile, sep='[ ]+', header=None, engine='python',names= ['sex', 'Weight', 'Minutes'])
        df = data[['Minutes', 'sex', 'Weight']]
        
        x = df.Weight[df['sex'] == 1].values
        y = df.Weight[df['sex'] == 2].values

        plt.hist(x,bins=25, label = 'sex1')
        plt.hist(y,bins=25,label = 'sex2')
        plt.legend(loc = 'upper right')
        plt.show()

def zad9c():
        inFile = 'http://ww2.amstat.org/publications/jse/datasets/babyboom.dat.txt'
        data = pd.read_csv(inFile, sep='[ ]+', header=None, engine='python',names= ['sex', 'Weight', 'Minutes'])
        df = data[['Minutes', 'sex', 'Weight']]
        
        x = df.Weight[df['sex'] == 1].values
        y = df.Weight[df['sex'] == 2].values

        sns.kdeplot(x,label = 'sex1')
        sns.kdeplot(y,label = 'sex2')

        plt.legend(loc = 'upper right')
        plt.show()

def zad9d():
        inFile = 'http://ww2.amstat.org/publications/jse/datasets/babyboom.dat.txt'
        data = pd.read_csv(inFile, sep='[ ]+', header=None, engine='python',names= ['sex', 'Weight', 'Minutes'])
        df = data[['Minutes', 'sex', 'Weight']]
        
        x = df.Weight[df['sex'] == 1].values
        y = df.Weight[df['sex'] == 2].values

        plt.plot(stats.cumfreq(x,numbins=25)[0],label = 'sex1')
        plt.plot(stats.cumfreq(y,numbins=25)[0],label = 'sex2')

        plt.legend(loc = 'upper right')
        plt.show()

def zad9e():
        inFile = 'http://ww2.amstat.org/publications/jse/datasets/babyboom.dat.txt'
        data = pd.read_csv(inFile, sep='[ ]+', header=None, engine='python',names= ['sex', 'Weight', 'Minutes'])
        df = data[['Minutes', 'sex', 'Weight']]
        
        x = df.Weight[df['sex'] == 1].values
        y = df.Weight[df['sex'] == 2].values

        plt.boxplot(x,sym='*',label = 'sex1')
        plt.boxplot(y,sym='*',label = 'sex2')

        plt.legend(loc = 'upper right')
        plt.show()

def zad9f():
        inFile = 'http://ww2.amstat.org/publications/jse/datasets/babyboom.dat.txt'
        data = pd.read_csv(inFile, sep='[ ]+', header=None, engine='python',names= ['sex', 'Weight', 'Minutes'])
        df = data[['Minutes', 'sex', 'Weight']]
        
        x = df.Weight[df['sex'] == 1].values
        y = df.Weight[df['sex'] == 2].values

        plt.violinplot(x)
        #plt.violinplot(y)

        plt.legend(loc = 'upper right')
        plt.show()


def zad10():
        mean = 0
        sigma = 1

        x = np.linspace(mean - 3*sigma, mean+3*sigma, 1000)
        y = stats.norm(loc = mean, scale = sigma).pdf(x) #probability density function

        c = 3 
        mask = (x >= mean - c) & (x <= mean + c)

        filled_x, filled_y = x[mask], y[mask]

        plt.plot(x,y)
        plt.fill_between(filled_x,filled_y, color = 'b')
        plt.show()

        print(np.trapezoid(filled_y,filled_x))

zad10()


