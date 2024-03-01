import functools

list = [1, 2, 3, 4, 5, 6, 7, 8]

print("Sum: ", end="")
print(functools.reduce(lambda a, b: a + b, list))

print("Min: ", end="")
print(functools.reduce(lambda a, b: b if b < a else a, list))

print("Prod: ", end="")
print(functools.reduce(lambda a, b: a * b, list))
