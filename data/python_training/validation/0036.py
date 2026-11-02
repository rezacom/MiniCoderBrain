def prime2(x):
    return x > 1 and all(x % y for y in range(2, x))
