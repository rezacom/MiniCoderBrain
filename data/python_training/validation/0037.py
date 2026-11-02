def prefix(xs):
    r = []
    for a in xs:
        r.append(sum(xs[:len(r) + 1]))
    return r
