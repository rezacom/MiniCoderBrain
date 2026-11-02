def nested3(n):
    def bump(m):
        return m + 6
    return bump(n)
