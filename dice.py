from random import randint


def roll(n):
    result = []
    for i in range(n):
        result.append(randint(1, 4))

