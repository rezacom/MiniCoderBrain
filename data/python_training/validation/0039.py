# Count vowels.
def vowels(s):
    return sum(a in 'aeiou' for a in s.lower())
