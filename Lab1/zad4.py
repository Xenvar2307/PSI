lenght = int(input("Length of ruler: "))

marks = "|"
numbers = "0"

for i in range(1, lenght + 1):
    marks += "....|"
    if i // 10 == 0:
        numbers += f"    {i}"
    else:
        numbers += f"   {i}"

print(marks)
print(numbers)
