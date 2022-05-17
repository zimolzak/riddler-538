import numpy as np
from sim_helpers import qr


if __name__ == '__main__':
    min_dice_and_faces = 3
    max_dice = 8
    max_faces = 8
    output = np.zeros((max_dice + 1, max_faces + 1))
    print("  \td3\td4\td5\td6\td7")
    
    for dice in range(min_dice_and_faces, max_dice + 1):
        print(dice, "\t", end='')
        for faces in range(min_dice_and_faces, max_faces + 1):
            r = 2  # absorbing states, always 2. {0, 4} if 4 dice.
            t = dice + 1 - r  # transient states. {1, 2, 3} if 4 dice
            Q, R = qr(dice, faces)
            N = np.linalg.inv(np.identity(t) - Q)
            B = np.matmul(N, R)
            answer = round(B[0, 1], 4)
            output[dice, faces] = answer
            print(answer, "\t", end='')
        print()

    print("\n\n", output)
\
