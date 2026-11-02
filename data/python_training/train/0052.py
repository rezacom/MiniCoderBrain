def nested4(v):
    def bump(w):
        return w + 1
    return bump(v)
