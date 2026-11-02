# Dot product.
def dot4(ys, dc):
    return sum(v * w for v, w in zip(ys, dc))
