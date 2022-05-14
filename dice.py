from random import randint
import numpy as np
from tqdm import tqdm
from itertools import product


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


def game_won(num_dice, num_faces, do_transitions=False, do_history=False):
    """note that "history" in return tuple is whole history of all dice values.
    """
    transitions = np.zeros((5, 5))  # fixme generalize numeric
    old_score = 1
    r = roll(num_dice, num_faces)
    n_rolls = 1
    history = []
    while True:
        u, d = parse_dice(r)

        score = len(u)
        if do_history:
            history.append([u, d])
        if do_transitions:
            transitions[old_score, score] += 1
        if score == 4:  # win
            if do_transitions:
                transitions[3, 4] += 1  # fixme generalize numeric
                transitions[4, 4] += 1  # fixme generalize numeric
            return True, n_rolls, history, transitions
        elif score == 0:  # loss
            if do_transitions:
                transitions[0, 0] += 1
            return False, n_rolls, history, transitions
        else:
            old_score = score
            r = reroll(u, d, num_faces)
            n_rolls += 1


def simulate_many(n_sims, num_dice, num_faces, do_transitions=False):
    # first game ({0,1} win, duration, history, transitions)
    w, n, _, transitions = game_won(num_dice, num_faces)
    games_matrix = np.array([[w, n]])

    for i in tqdm(range(n_sims - 1)):
        w, n, _, ti = game_won(num_dice, num_faces, do_transitions=do_transitions)
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
    m, tr = simulate_many(n_sims_setting, num_dice=ndi, num_faces=nfa, do_transitions=True)

    win_loss = m[:, 0]
    durations = m[:, 1]
    n_wins = np.sum(win_loss)
    dur_win = durations[win_loss == 1]
    dur_loss = durations[win_loss == 0]
    p_numeric = np.mean(win_loss)

    print()
    print("P =", p_numeric)  # P = 0.45132 at 200 k
    print()

    print("N trials =", n_sims_setting)
    print("max game dur overall:", np.max(durations))
    print("length checks (wins and losses):")
    print(" ", np.sum(win_loss), np.shape(dur_win))
    print(" ", n_sims_setting - n_wins, np.shape(dur_loss))
    print("max dur win vs loss:", np.max(dur_win), np.max(dur_loss))
    print("tr =\n", tr)
    print()




    print("\n\n\n# Absorbing Markov chain calculations\n")

    """We have a Markov chain. tr_normalized is a right stochastic matrix. S is state space {0,1,2,3,4}
    with cardinality alpha = 5. Absorbing states are 0 and 4."""

    # Search in space of all possible 4-dice rolls.
    # Count how many unique dice.
    # Equivalent to finding probability of (2 pair + 4 of kind), 3 of kind, pair, straight.
    score_4 = [0, 0, 0, 0, 0]  # fixme generalize
    for four_dice_tup in product(range(1, nfa + 1), repeat=ndi):
        uniq, dupe = parse_dice(list(four_dice_tup))
        score = len(uniq)
        score_4[score] += 1
    print("Closed-form transition vector for init or 3 dups, 1 unique")
    print("(same as row 1 or 5 of Transition matrix):")
    print(score_4, "/ 256 =")
    cf = [x / 256 for x in score_4]  # fixme generalize
    print(cf)
    print()

    """Absorbing Markov chain:
    t = 3 transient states {1, 2, 3}
    r = 2 absorbing states {0, 4}
    Q is 3x3
    R is 3x2
    
    Canonical form transition matrix P =
    q q q r r
    q q q r r
    q q q r r
    0 0 0 1 0
    0 0 0 0 1
    """

    r = 2
    t = ndi + 1 - r
    # fixme generalize Q and R, using cf[] more.
    Q = np.array([
        [cf[1], cf[2], cf[3]],  # transition from 1 to something, denom 256
        [1/8, 5/8, 0],  # from 2 to something
        [0, 0, 0]  # hopefully works
    ])
    R = np.array([
        [cf[0], cf[4]],
        [1/8, 1/8],
        [0, 1]
    ])
    print("Q = transient to transient =")
    print(Q)
    print("R = transient to absorbing =")
    print(R)
    print()

    # fundamental matrix
    N = np.linalg.inv(np.identity(t) - Q)
    print("N = (I - Q)^-1 = ")
    print(N)
    print()

    # Absorbing probabilities (start in trans state i, land in abs state j)
    # Remember, the whole game starts basically in trans state 1 (same as 1 unique, 3 dup)
    # So Pr(win) = B_11 = 0.45, agreeing fine with numeric monte carlo estimate.
    # Pr(loss) = B_10 = 0.55.
    # Here, Python would call it B[0, 1] but I am calling it B_11
    B = np.matmul(N, R)
    print("B = N * R =")
    print(B)
    print()

    print("Reminder, numeric monte carlo estimate was:")
    print(p_numeric)
    print()

    print("FINAL ANSWER!")
    print(B[0, 1])
    print()
