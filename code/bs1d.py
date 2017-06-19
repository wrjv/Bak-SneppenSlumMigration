#
# Bax-Sneppen 1D implementation by
# Nicky, Maarten, Wessel and Willem
#

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


class BaxSneppen1D(object):

    def __init__(self, initial_values):
        self.states = [initial_values]
        self.ages = [np.zeros(len(initial_values))]
        self.time = 0

        self.update_freq = 100

    def execute(self):
        while self.update_state():
            continue

        print(self.ages[-1])
        print(len(self.states))

    def update_state(self):
        new_ages = deepcopy(self.ages[-1])

        # Build a new state
        new_state = deepcopy(self.states[-1])

        for i in range(self.update_freq):
            new_ages = new_ages + 1
            min_val = np.argmin(new_state)

            # Stopping criterium
            #if new_state[min_val] > 0.67:
            #    return False
            if len(self.states) > 100000:
                return False

            # Modify the values around the minimum value
            new_state[min_val - 1] = np.random.uniform(0, 1, 1)
            new_state[min_val] = np.random.uniform(0, 1, 1)
            new_state[(min_val + 1) % len(new_state)] = np.random.uniform(0, 1, 1)

            # Modify the cell ages
            new_ages[min_val - 1] = 0
            new_ages[min_val] = 0
            new_ages[(min_val + 1) % len(new_state)] = 0

        self.states.append(new_state)
        self.ages.append(new_ages)
        self.time += 1
        return True

    def plot_ages(self):
        plt.imshow(self.ages[::100], aspect='auto', cmap='jet_r', interpolation='nearest')
        plt.show()


def main():
    initial_values = np.random.rand(4000)
    bs1d = BaxSneppen1D(initial_values)
    bs1d.execute()
    bs1d.plot_ages()

if __name__ == '__main__':
    main()
