ruler_lenght = int(input("Length of ruler: "))
if ruler_lenght > 20:
    ruler_lenght = 20

marks = "|"
numbers = "0"

next_mark = "....|"

for i in range(1, ruler_lenght + 1):
    marks += "....|"
    numbers += " "*(len(next_mark) - len(str(i))) + str(i)

print(marks)
print(numbers)
