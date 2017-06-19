#
# Bax-Sneppen 1D implementation by
# Nicky, Maarten, Wessel and Willem
#

import numpy as np

from copy import deepcopy


class BaxSneppen1D(object):

    def __init__(self, initial_values):
        self.states = [initial_values]
        self.ages = [np.zeros(len(initial_values))]

    def execute(self):
        while self.update_state():
            continue

        print(self.ages[-1])
        print(len(self.states))

    def update_state(self):
        self.ages.append(self.ages[-1] + 1)

        # Build a new state
        new_state = deepcopy(self.states[-1])

        min_val = np.argmin(new_state)

        # Stopping criterium
        if new_state[min_val] > 0.60:
            return False

        # Modify the values around the minimum value
        new_state[min_val - 1] = np.random.uniform(0, 1, 1)
        new_state[min_val] = np.random.uniform(0, 1, 1)
        new_state[(min_val + 1) % len(new_state)] = np.random.uniform(0, 1, 1)

        # Modify the cell ages
        self.ages[-1][min_val - 1] = 0
        self.ages[-1][min_val] = 0
        self.ages[-1][(min_val + 1) % len(new_state)] = 0

        self.states.append(new_state)

        return True


def main():
    initial_values = np.random.rand(100)
    bs1d = BaxSneppen1D(initial_values)
    bs1d.execute()

if __name__ == '__main__':
    main()
