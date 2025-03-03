import random

random.seed(4)

number_of_tests = 1000000 #milion
count = 0

for i in range(number_of_tests):
    x = random.random()
    y = random.random()
    if x**2 + y**2 <= 1:
        count += 1

print(f"Estimated Pi: {4*count/number_of_tests}")
