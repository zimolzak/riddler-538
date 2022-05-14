import numpy as np
from itertools import product
from sim_helpers import simulate_many, parse_dice


if __name__ == '__main__':

    # global settings, might need in multiple places if generalize
    ndi = 4
    nfa = 4

    n_sims_setting = 50000
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
        sc = len(uniq)
        score_4[sc] += 1
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
