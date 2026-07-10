# simple_error.py

name = input("Enter your name: ")
age = input("Enter your age: ")

print("Hello " + name)

# Fix 1: age is a string, convert to int before adding
next_year = int(age) + 1

print("Next year you will be", next_year)

numbers = [10, 20, 30]

# Fix 2: List index out of range - use valid index (0, 1, 2)
print("Fourth number:", numbers[2])

# Fix 3: Variable name typo - 'total' not 'totall'
total = 100
print("Total is", total)

# Fix 4: Division by zero - use a non-zero divisor
result = 100 / 1

print("Result:", result)