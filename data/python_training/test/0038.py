# Count odd numbers.
def odd_count2(ns):
    return sum(x % 2 != 0 for x in ns)
