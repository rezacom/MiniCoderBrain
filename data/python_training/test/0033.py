def find(xs, a):
    for b, c in enumerate(xs):
        if c == a:
            return b
    return -1
