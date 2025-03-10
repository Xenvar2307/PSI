R = int(input("Number of Rows: "))
C = int(input("Number of Columns: "))

row_start = 1
printed_value = row_start

for r in range(R):
    row = ""
    printed_value = row_start

    for c in range(C):
        row = row + str(printed_value)
        printed_value = 0 if printed_value == 1 else 1

    row_start = 0 if row_start == 1 else 1

    print(row)
