import numpy as np
import scipy.stats as st
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
from scipy import optimize



def zad1():
    f = lambda x: (x**2)
    x = np.array([.2, .5, .8, .9, 1.3, 1.7, 2.1, 2.7])
    y = f(x) + np.random.randn(len(x))
    deg = 1
    A = np.vander(x, deg + 1)
    A_T = A.transpose()
    L = np.linalg.inv(A_T @ A) @ A_T @ y
    print(L)

def zad2():
    def f(x, *args):
        u,v = x
        return (u+1)**2 + v**2
    
    def gradf(x, *args):
        u,v = x
        gu = 2*u + 2
        gv = 2*v
        return np.asarray((gu,gv))

    x0 = np.asarray((1,1))
    optimize.fmin_cg(f,x0)
    optimize.fmin_cg(f,x0, fprime=gradf)

def zad3_4():
    f = lambda x: (x**2)
    x = np.array([.2, .5, .8, .9, 1.3, 1.7, 2.1, 2.7])
    y = f(x) + np.random.randn(len(x))
    def compute_error(x, *args):
        a,b = x
        X,Y = args
        return np.sum(np.square(Y - (X*a + b)))

    print("Zad3 test:",compute_error((2,1), x, y))

    x0 = np.asarray((2,-2))
    print("Zad4: ", optimize.fmin_cg(compute_error, x0, args=(x,y)))

    deg = 1
    A = np.vander(x, deg + 1)
    A_T = A.transpose()
    L = np.linalg.inv(A_T @ A) @ A_T @ y
    print("From zad1: ",L)

def zad5_6():
    f = lambda x: (x**2)
    x = np.array([.2, .5, .8, .9, 1.3, 1.7, 2.1, 2.7])
    y = f(x) + np.random.randn(len(x))
    def compute_error(x, *args):
        a,b = x
        X,Y = args
        return np.sum(np.abs(Y - (X*a + b)))

    print("Zad3 test:",compute_error((2,1), x, y))

    x0 = np.asarray((2,-2))
    print("Zad4: ", optimize.fmin_cg(compute_error, x0, args=(x,y)))

    deg = 1
    A = np.vander(x, deg + 1)
    A_T = A.transpose()
    L = np.linalg.inv(A_T @ A) @ A_T @ y
    print("From zad1: ",L)


def zad7():
    def l(X, m, d):
        f = st.norm(m,d).pdf
        return np.sum(np.log(f(X)))
    
    x = np.array([.2, .5, .8, .9, 1.3, 1.7, 2.1, 2.7])
    mean = 0
    cov = 1

    print(l(x, mean, cov))

def zad8():
    def SplitGaussianDensityFunction(x, m,d,r):
        result = []
        c = np.sqrt(2/np.pi) * d**(-1/2) / (1 + r)
        for x in x:
            #print(result)
            if ( x <= m ):
                result.append(c * np.exp(-1 * (x-m)**2 / (2*d) ))
            else:
                result.append(c * np.exp(-1 * (x-m)**2 / (2*d*r**2) ))
        return np.array(result)

    x = np.array(np.linspace(-5, 5, 200))
    y_1 = SplitGaussianDensityFunction(x, 0, 1, 1)
    y_2 = SplitGaussianDensityFunction(x, 0, 1, 1/2)
    y_3 = SplitGaussianDensityFunction(x, 1, 1/4, 1)

    plt.plot(x,y_1, label="a")
    plt.plot(x,y_2, label="b")
    plt.plot(x,y_3, label="c")
    plt.legend(loc = 'upper right')

    plt.show()


zad8()