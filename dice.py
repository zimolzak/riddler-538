from random import randint
import numpy as np
from tqdm import tqdm
from itertools import product

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
    """note that "history" in return tuple is whole history of all dice values.

    :param num_dice:
    :param num_faces:
    :return:
    """
    transitions = np.zeros((6, 5))  # 10 11 12 14 20 21 22 24
    old_score = 5
    # Use row 5 to tally initialization weights. Others are ordinary meaning.
    r = roll(num_dice, num_faces)
    n_rolls = 1
    history = []
    while True:
        u, d = parse_dice(r)

        score = len(u)
        history.append([u, d])
        transitions[old_score, score] += 1

        if score == 4:  # win
            transitions[3, 4] += 1
            transitions[4, 4] += 1
            return True, n_rolls, history, transitions
        elif score == 0:  # loss
            transitions[0, 0] += 1
            return False, n_rolls, history, transitions
        else:
            old_score = score
            r = reroll(u, d, num_faces)
            n_rolls += 1


def demonstrate_one_game():  # takes no params
    res, n, hist, _ = game_won(4, 4)  # just uses sensible defaults down here
    print(res, n)
    print()
    for row in hist:
        print(row[0], row[1])


def simulate_many(n_sims, num_dice, num_faces):
    # first game ({0,1} win, duration, history, transitions)
    w, n, _, transitions = game_won(num_dice, num_faces)
    games_matrix = np.array([[w, n]])

    for i in tqdm(range(n_sims - 1)):
        w, n, _, ti = game_won(num_dice, num_faces)
        games_matrix = np.vstack((
            games_matrix,
            [[w, n]]
        ))
        transitions += ti
    return games_matrix, transitions


if __name__ == '__main__':

    # global settings, might need in multiple places if generalize
    ndi = 4
    nfa = 4

    n_sims_setting = 20000
    m, tr = simulate_many(n_sims_setting, num_dice=ndi, num_faces=nfa)

    win_loss = m[:, 0]
    durations = m[:, 1]
    n_wins = np.sum(win_loss)
    dur_win = durations[win_loss == 1]
    dur_loss = durations[win_loss == 0]

    print()
    print("P =", np.mean(win_loss))  # P = 0.45132 at 200 k
    print()

    print("N trials =", n_sims_setting)
    print("max game dur overall:", np.max(durations))
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
    print("Shape of intermediate tensor for 'extra credit,' or eventually")
    print("doing histogram of duration stratified by win/loss:")
    print(t.shape)
    print("Expect (2, max dur, n sims)")

    tr_rowsum = tr.sum(axis=1).reshape((tr.shape[0], 1))
    tr_allsum = np.broadcast_to(tr_rowsum, tr.shape)
    tr_normalized = tr / tr_allsum
    print()

    print("Transition matrix:")
    # print(tr, "\n")
    # print(tr_allsum, "\n")
    print(tr_normalized, "\n")

    """We have a Markov chain. tr_normalized is a right stochastic matrix. S is state space {0,1,2,3,4}
    with cardinality alpha = 5. Absorbing states are 0 and 4."""

    # Search in space of all possible 4-dice rolls.
    # Count how many unique dice.
    # Equivalent to finding probability of (2 pair + 4 of kind), 3 of kind, pair, straight.
    score_4 = [0, 0, 0, 0, 0]
    for four_dice_tup in product(range(1, nfa + 1), repeat=ndi):
        uniq, dupe = parse_dice(list(four_dice_tup))
        score = len(uniq)
        score_4[score] += 1
    print("Closed-form transition vector for init or 3 dups")
    print("(same as row 1 or 5 of Transition matrix):")
    print(score_4, "/ 256 =")
    print([x / 256 for x in score_4])

    """Absorbing Markov chain:
    t = 3 transient states
    r = 2 absorbing states
    Q is 3x3
    R is 3x2
    
    Canonical form transition matrix P =
    q q q r r
    q q q r r
    q q q r r
    0 0 0 1 0
    0 0 0 0 1
    """
