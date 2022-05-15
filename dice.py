import numpy as np
from sim_helpers import simulate_many, search_space, qr


if __name__ == '__main__':

    # Global settings, might need in multiple places if generalize
    ndi = 4
    nfa = 6
    n_sims_setting = 10000

    # Numeric
    print("# Monte carlo simulation\n")
    m, tr = simulate_many(n_sims_setting, num_dice=ndi, num_faces=nfa, do_transitions=True)
    win_loss = m[:, 0]
    n_wins = np.sum(win_loss)
    p_numeric = np.mean(win_loss)
    print()
    print("P =", p_numeric)  # P = 0.45132 at 200 k
    print()
    print("N trials =", n_sims_setting)
    print("tr =")
    for row in tr:
        print(row / np.sum(row))
    print()

    # Markov
    print("# Absorbing Markov chain calculations\n")
    score_4, cf = search_space(ndi, nfa, ndi)
    print(score_4, "\n", cf)
    print()

    # Will need to do these calcs for the following parts of transition matrix:
    # NOT row 0 (absorbing); set it to [1 0 0 0 ...] --> ignore, doesn't go into Q.
    # NOT row ndi-1 (never happens) or ndi (absorbing). Set them to [... 0 0 0 1] --> ignore ndi.
    # Yes for rows 1 through ndi-2 (1 to 2 if tetrahedral)
    # So: stack up 1 through ndi-1
    # Cut out middle and that's Q.
    # Cut out left and right and that's R.

    r = 2  # absorbing states, always 2. {0, 4} if 4 dice.
    t = ndi + 1 - r  # transient states. {1, 2, 3} if 4 dice

    Q, R = qr(ndi, nfa)
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

    B = np.matmul(N, R)
    print("B = N * R =")
    print(B)
    print()

    print("Reminder, numeric monte carlo estimate was:")
    print(p_numeric)
    print()

    # Pr(win) = B[0, 1] = 0.45, agreeing fine with numeric monte carlo estimate.
    print("FINAL ANSWER!")
    print(B[0, 1])
    print()
