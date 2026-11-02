def prime(a):
    return a > 1 and all(a % b for b in range(2, a))
