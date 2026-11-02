def smallest4(ys):
    q = ys[0]
    for v in ys:
        if v < q:
            q = v
    return q
