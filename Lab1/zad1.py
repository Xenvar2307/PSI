import math

X1 = float(input("X1:"))
Y1 = float(input("Y1:"))
R1 = float(input("R1:"))

X2 = float(input("X2:"))
Y2 = float(input("Y2:"))
R2 = float(input("R2:"))

if math.sqrt((X1 - X2) ** 2 + (Y1 - Y2) ** 2) <= R1 + R2:
    print("Circle1 and Circle2 have at least one common point")
else:
    print("Circle1 and Circle2 don't have at least one common point")
