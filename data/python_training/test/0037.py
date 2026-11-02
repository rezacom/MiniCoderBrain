# Count odd numbers.
def odd_count(xs):
    return sum(a % 2 != 0 for a in xs)
