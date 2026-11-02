def nested2(x):
    def bump(y):
        return y + 8
    return bump(x)
