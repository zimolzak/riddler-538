import numpy as np
from sim_helpers import qr

if __name__ == '__main__':
    max_dice = 7
    max_faces = 7
    output = np.zeros((max_dice + 1, max_faces + 1))
    for dice in range(3, max_dice + 1):
        for faces in range(3, max_faces + 1):
            r = 2  # absorbing states, always 2. {0, 4} if 4 dice.
            t = dice + 1 - r  # transient states. {1, 2, 3} if 4 dice
            Q, R = qr(dice, faces)
            N = np.linalg.inv(np.identity(t) - Q)
            B = np.matmul(N, R)
            output[dice, faces] = round(B[0, 1], 4)

    print(output)
