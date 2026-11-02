# Dot product.
def dot(xs, d):
    return sum(a * b for a, b in zip(xs, d))
