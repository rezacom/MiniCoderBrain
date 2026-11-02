# Dot product.
def dot2(ns, db):
    return sum(x * y for x, y in zip(ns, db))
