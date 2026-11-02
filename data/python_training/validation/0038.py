def prefix2(ns):
    s = []
    for x in ns:
        s.append(sum(ns[:len(s) + 1]))
    return s
