#
# Bax-Sneppen 2D implementation by
# Nicky, Maarten, Wessel and Willem
#

import numpy as np

from copy import deepcopy


class BaxSneppen2D(object):

    def __init__(self, initial_values):
        self.states = [initial_values]
        self.ages = [np.zeros((len(initial_values), len(initial_values[0])))]

    def execute(self, moore=False):
        while self.update_state(moore):
            continue

        print(self.ages[-1])
        print(len(self.states))

    def update_state(self, moore=False):
        new_ages = self.ages[-1] + 1

        # Build a new state
        new_state = deepcopy(self.states[-1])

        min_val = np.argmin(new_state)
        y = min_val // len(new_state)
        x = min_val % len(new_state[0])
        # Stopping criterium
        if new_state[y][x] > 0.205:
            return False

        # Modify the values around the minimum value
        new_state[y][x] = np.random.uniform(0, 1, 1)
        new_state[y - 1][x] = np.random.uniform(0, 1, 1)
        new_state[y][x - 1] = np.random.uniform(0, 1, 1)
        new_state[(y + 1) % len(new_state)][x] = np.random.uniform(0, 1, 1)
        new_state[y][(x + 1) % len(new_state[0])] = np.random.uniform(0, 1, 1)

        # Modify the cell ages
        new_ages[y][x] = 0
        new_ages[y - 1][x] = 0
        new_ages[(y + 1) % len(new_state)][x] = 0
        new_ages[y][x - 1] = 0
        new_ages[y][(x + 1) % len(new_state[0])] = 0

        if moore:
            new_state[y - 1][x - 1] = np.random.uniform(0, 1, 1)
            new_state[(y + 1) % len(new_state)][(x + 1) % len(new_state[0])] = np.random.uniform(0, 1, 1)
            new_state[(y + 1) % len(new_state)][x - 1] = np.random.uniform(0, 1, 1)
            new_state[y - 1][(x + 1) % len(new_state[0])] = np.random.uniform(0, 1, 1)

            new_ages[y - 1][x - 1] = 0
            new_ages[(y + 1) % len(new_state)][(x + 1) % len(new_state[0])] = 0
            new_ages[(y + 1) % len(new_state)][x - 1] = 0
            new_ages[y - 1][(x + 1) % len(new_state[0])] = 0

        self.states.append(new_state)
        self.ages.append(new_ages)

        return True


def main():
    initial_values = np.random.rand(50, 50)
    bs2d = BaxSneppen2D(initial_values)
    bs2d.execute(True)

if __name__ == '__main__':
    main()
