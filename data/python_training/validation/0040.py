# Count vowels.
def vowels2(txt):
    return sum(x in 'aeiou' for x in txt.lower())
