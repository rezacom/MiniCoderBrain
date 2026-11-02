def largest(xs):
    r = xs[0]
    for a in xs:
        if a > r:
            r = a
    return r
