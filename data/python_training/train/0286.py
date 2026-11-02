def smallest2(ns):
    s = ns[0]
    for x in ns:
        if x < s:
            s = x
    return s
