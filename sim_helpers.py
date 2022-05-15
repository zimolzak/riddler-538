from random import randint
from tqdm import tqdm
import numpy as np
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


def search_space(ndi, nfa, ndi_total):
    """Search in space of all possible 4-dice rolls.
    (But not necessarily 4)
    Count how many unique dice in each element of the space.
    Equivalent to finding probability of (2 pair + 4 of kind), 3 of kind, pair, straight.
    """
    score_vec = np.zeros(ndi_total + 1)
    if ndi == 1:
        # Handle weird case that never happens. Pretend it's transient to state ndi_total.
        score_vec[-1] = nfa ** ndi
        return score_vec, score_vec / (nfa ** ndi)  # [... 0 0 0 0 1]
    for few_dice_tup in product(range(1, nfa + 1), repeat=ndi):
        n_arb = ndi_total - ndi
        arbitrary_dice = list(range(1, n_arb + 1))  # The ones in "unique" group. Numbers don't matter.
        all_dice = list(few_dice_tup) + arbitrary_dice
        assert len(all_dice) == ndi_total
        uniq, dupe = parse_dice(all_dice)
        sc = len(uniq)
        score_vec[sc] += 1
    return score_vec, score_vec / (nfa ** ndi)


def ordinary_transition_matrix(ndi, nfa):
    """Not square with rows 0 to ndi, but rows 1 to ndi-1."""
    _, result = search_space(ndi, nfa, ndi)  # row 1 means roll ALL dice. Should be equiv to roll all-1.
    # But must only do one. 10 8 7 6 5 4 3 2 1. Note the skip. NOT 10 9 8 7 6 ....
    for i in range(ndi - 2, 0, -1):
        if i == 1:
            row_i = np.zeros(result.shape[1])
        else:
            _, row_i = search_space(i, nfa, ndi)
        result = np.vstack((result, row_i))
    return result
