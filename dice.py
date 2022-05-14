from random import randint
import numpy as np
from tqdm import tqdm

ADJACENCY = np.array([
    # Edges in directed graph of "score" over time.
    # Score ranges from 0 to 4 (number of dice in the "unique" group).
    # Only 8 can happen in a real game.
    # Each would have a weight.
    # Making {0,1} matrix of the 8 allowed edges in case we need it later.
    [0, 0, 0, 0, 0],
    [1, 1, 1, 0, 1],
    [1, 1, 1, 0, 1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])


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
        history.append([u, d])
        if len(d) == 0:
            return True, n_rolls, history
        elif len(u) == 0:
            return False, n_rolls, history
        else:
            r = reroll(u, d, num_faces)
            n_rolls += 1


def demonstrate_one_game():  # takes no params
    res, n, hist = game_won(4, 4)  # just uses sensible defaults down here
    print(res, n)
    print()
    for row in hist:
        print(row[0], row[1])


def simulate_many(n_sims, num_dice, num_faces):
    w, n, _ = game_won(num_dice, num_faces)

    games_matrix = np.array([[w, n]])
    for i in tqdm(range(n_sims - 1)):
        w, n, _ = game_won(num_dice, num_faces)
        games_matrix = np.vstack((
            games_matrix,
            [[w, n]]
        ))
    return games_matrix


if __name__ == '__main__':
    n_sims_setting = 20000
    m = simulate_many(n_sims_setting, 4, 4)

    win_loss = m[:, 0]
    durations = m[:, 1]
    n_wins = np.sum(win_loss)
    dur_win = durations[win_loss == 1]
    dur_loss = durations[win_loss == 0]
    transitions = np.zeros(8)  # 10 11 12 14 20 21 22 24

    print()
    print("P =", np.mean(win_loss))  # P = 0.45132 at 200 k
    print()

    print("max dur global:", np.max(durations))
    print("length checks (wins and losses):")
    print(" ", np.sum(win_loss), np.shape(dur_win))
    print(" ", n_sims_setting - n_wins, np.shape(dur_loss))
    print("max dur win vs loss:", np.max(dur_win), np.max(dur_loss))
    print()

    # build up tensor by tiling
    dw = np.multiply(durations, win_loss)
    dl = np.multiply(durations, 1 - win_loss)
    t = np.stack((
        np.broadcast_to(dw, (np.max(durations), n_sims_setting)),
        np.broadcast_to(dl, (np.max(durations), n_sims_setting))
    ))
    print("Shape of intermediate tensor for eventually")
    print("doing histogram of duration stratified by win/loss:")
    print(t.shape)
    print("Expect (2, max dur, n sims)")
