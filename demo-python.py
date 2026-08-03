# simple_error.py — fixed version

name = input("Enter your name: ")
age = input("Enter your age: ")

print("Hello " + name)

# Fix 1: Convert age to int before adding
next_year = int(age) + 1

print("Next year you will be", next_year)

numbers = [10, 20, 30]

# Fix 2: Access index 2 (third element) instead of index 3 (out of range)
print("Fourth number:", numbers[2])

# Fix 3: Corrected variable name from 'totall' to 'total'
total = 100
print("Total is", total)

# Fix 4: Divide by non-zero value
result = 100 / 10

print("Result:", result)
