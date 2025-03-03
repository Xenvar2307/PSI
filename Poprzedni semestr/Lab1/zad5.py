import functools

example_list = [1, 2, 3, 4, 5, 6, 7, 8]

def sum(list):
    print("Sum: ", end="")
    print(functools.reduce(lambda a, b: a + b, list))

def min(list):
    print("Min: ", end="")
    print(functools.reduce(lambda a, b: b if b < a else a, list))

def prod(list):
    print("Prod: ", end="")
    print(functools.reduce(lambda a, b: a * b, list))


