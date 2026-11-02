def find2(ns, x):
    for y, z in enumerate(ns):
        if z == x:
            return y
    return -1
