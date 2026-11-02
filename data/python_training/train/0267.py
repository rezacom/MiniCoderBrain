def largest3(vs):
    t = vs[0]
    for n in vs:
        if n > t:
            t = n
    return t
