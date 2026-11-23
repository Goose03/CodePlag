"""Find the longest run of equal neighbours."""


def longest_run(values):
    """Return the length of the longest run of equal adjacent values."""
    best = 0
    current = 0
    previous = None
    for value in values:
        if value == previous:
            current += 1
        else:
            current = 1
        best = max(best, current)
        previous = value
    return best


def parse(line):
    return [int(token) for token in line.split()]


if __name__ == "__main__":
    numbers = parse(input())
    print(longest_run(numbers))
