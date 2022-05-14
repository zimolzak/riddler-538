from random import randint
from tqdm import tqdm
import numpy as np


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
    transitions = np.zeros((num_dice + 1, num_dice + 1))
    old_score = 1
    my_roll = roll(num_dice, num_faces)
    n_rolls = 1
    history = []
    while True:
        u, d = parse_dice(my_roll)

        score = len(u)
        if do_history:
            history.append([u, d])
        if do_transitions:
            transitions[old_score, score] += 1
        if score == 4:  # win
            if do_transitions:
                transitions[num_dice - 1, num_dice] += 1  # The "never" transition like 3 -> 4 if tetrahedral
                transitions[num_dice, num_dice] += 1  # Win is an absorbing state.
            return True, n_rolls, history, transitions
        elif score == 0:  # loss
            if do_transitions:
                transitions[0, 0] += 1  # Win is an absorbing state.
            return False, n_rolls, history, transitions
        else:
            old_score = score
            my_roll = reroll(u, d, num_faces)
            n_rolls += 1


def simulate_many(n_sims, num_dice, num_faces, do_transitions=False):
    # first game ({0,1} win, duration, history, transitions)
    w, n, _, transitions = game_won(num_dice, num_faces)
    games_matrix = np.array([[w, n]])

    for _ in tqdm(range(n_sims - 1)):
        w, n, _, ti = game_won(num_dice, num_faces, do_transitions=do_transitions)
        games_matrix = np.vstack((
            games_matrix,
            [[w, n]]
        ))
        transitions += ti
    return games_matrix, transitions
