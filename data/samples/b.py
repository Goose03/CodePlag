# my solution
def max_streak(arr):
    res = 0   # answer
    cnt = 0
    last = None

    for x in arr:
        if x == last:
            cnt += 1
        else:
            cnt = 1
        res = max(res, cnt)
        last = x
    return res

def read_numbers(s):
    # convert the input
    return [int(t) for t in s.split()]

if __name__ == "__main__":
    data = read_numbers(input())
    print(max_streak(data))
