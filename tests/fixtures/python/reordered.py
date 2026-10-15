def parse(line):
    return [int(token) for token in line.split()]


def longest_run(values):
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


if __name__ == "__main__":
    numbers = parse(input())
    print(longest_run(numbers))
