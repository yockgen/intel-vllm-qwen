# demo-python.py - Fixed version with proper error handling

name = input("Enter your name: ")
try:
    age = int(input("Enter your age: "))
except ValueError:
    print("Invalid age. Using 0 as default.")
    age = 0

print("Hello " + name)
next_year = age + 1
print("Next year you will be", next_year)

numbers = [10, 20, 30, 40]
try:
    print("Fourth number:", numbers[3])
except IndexError:
    print("Fourth number not available.")

total = 100
print("Total is", total)

# Division by zero - handle gracefully
try:
    result = 100 / 0
except ZeroDivisionError:
    result = 0
    print("Warning: Division by zero detected. Result set to 0.")

print("Result:", result)
