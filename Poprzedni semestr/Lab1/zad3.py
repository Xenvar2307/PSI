R = int(input("Number of Rows: "))
C = int(input("Number of Columns: "))

printed_value = 1

for r in range(R):
    row = ""

    for c in range(C):
        row = row + str(printed_value)
        printed_value = 0 if printed_value == 1 else 1

    print(row)
