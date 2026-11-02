def nested(a):
    def bump(b):
        return b + 7
    return bump(a)
