# If-else: two possibilities

fuel = 30
minimum = 50

if fuel >= minimum:
    print("Status: CLEARED FOR DEPARTURE")
    print("Fuel enough for travelling")
else:
    print("Status: HOLD START")
    print("Insufficient gasoline!")
    needed = minimum - fuel
    print(f"Need: {needed} more units")

# Access check
access_level = 2

if access_level >= 3:
    print("Bridge access: GRANTED")
else:
    print("Bridge access: DENIED")
