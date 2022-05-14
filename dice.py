from random import randint


def roll(num_dice, num_faces):
    result = []
    for i in range(num_dice):
        result.append(randint(1, num_faces))
    return result


def reroll(uniqs, dups, num_faces):
    rolled = roll(len(dups), num_faces)
    return uniqs + rolled


def parse_dice(dice: list) -> tuple:
    uniqs = []
    dups = []

    for value in range(max(dice) + 1):
        if dice.count(value) == 0:
            continue
        elif dice.count(value) == 1:
            uniqs.append(value)
            dice.remove(value)
        else:
            for i in range(dice.count(value)):
                dups.append(value)
                dice.remove(value)

    return uniqs, dups


def game_won(num_dice, num_faces):
    r = roll(num_dice, num_faces)
    n_rolls = 1
    history = []
    while True:
        u, d = parse_dice(r)
        history.append([u,d])
        if len(d) == 0:
            return True, n_rolls, history
        elif len(u) == 0:
            return False, n_rolls, history
        else:
            r = reroll(u, d, num_faces)
            n_rolls += 1


if __name__ == '__main__':
    res, n, hist = game_won(4, 4)
    print(res, n)
    print()
    for row in hist:
        print(row[0], row[1])
