import numpy as np
from sim_helpers import simulate_many, qr

if __name__ == '__main__':
    # Global settings
    ndi = 4
    nfa = 4
    n_sims_setting = 100000
    monte = True

    if monte:
        m, tr = simulate_many(n_sims_setting, num_dice=ndi, num_faces=nfa, do_transitions=True)
        win_loss = m[:, 0]
        p_numeric = np.mean(win_loss)
    else:
        p_numeric = None
        tr = None

    # Markov
    r = 2  # absorbing states, always 2. {0, 4} if 4 dice.
    t = ndi + 1 - r  # transient states. {1, 2, 3} if 4 dice
    Q, R = qr(ndi, nfa)
    N = np.linalg.inv(np.identity(t) - Q)
    B = np.matmul(N, R)

    print("%i dice, %i faces per die.\n" % (ndi, nfa))

    if monte:
        print("Transition matrix from simulation:")
        for row in tr:
            print(row / np.sum(row))
        print()

    print("Q = transient to transient =\n", Q, "\n")
    print("R = transient to absorbing =\n", R, "\n")
    print("B = N * R =\n", B, "\n")
    if monte:
        print(p_numeric, "Monte carlo estimate of win probability")
    print(B[0, 1], "FINAL ANSWER win probability!")
