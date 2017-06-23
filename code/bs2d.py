#
# Bax-Sneppen 2D implementation by
# Nicky, Maarten, Wessel and Willem
#
import itertools

from copy import deepcopy
from scipy.stats import multivariate_normal

import numpy as np


class BaxSneppen2D(object):
    def __init__(self, slum_size=(15, 15), empty_percent=0.3, cell_decrease_factor=0.8):
        self.state = np.ones(slum_size) * 2
        self.ages = np.ones(slum_size) * -1
        if empty_percent != 1:
            self.populate(empty_percent, slum_size)

        self.cur_av_count = 0
        self.cur_av_start = -1
        self.size = slum_size
        self.cell_decrease_factor = cell_decrease_factor
        xmean = self.size[0]*0.5
        ymean = self.size[1]*0.5
        cov = np.array([[xmean*0.8, 0], [0, ymean*0.8]])
        self.mvn = multivariate_normal([xmean, ymean], cov)

    def populate(self, empty_percent, slum_size):
        for x in range(slum_size[0]):
            for y in range(slum_size[1]):
                self.state[y][x] = np.random.uniform(0, 1, 1)
                self.ages[y][x] = 0

        empty = np.random.choice(range(slum_size[0] * slum_size[1]),
                                 empty_percent * slum_size[0] * slum_size[1], replace=False)
        for i in empty:
            self.state[i % slum_size[0]][i // slum_size[0]] = 2
            self.ages[i % slum_size[0]][i // slum_size[0]] = -1

    def get_min_val(self):
        return np.min(self.state)

    def get_min_val_index(self):
        return np.argmin(self.state)

    def get_avg_val(self):
        return np.average([i for i in self.state.flatten() if i != 2])

    def has_empty(self):
        empty = np.where(self.state == 2)

        if len(empty[0]) == 0:
            return False

        return True

    def add_to_grid(self):
        empty = np.where(self.state == 2)
        if len(empty[0]) == 0:
            return False

        es = [(x, y) for x, y in zip(empty[0], empty[1])]
        pvalues = np.array([self.mvn.pdf([e[0], e[1]]) for e in es])
        pvalues /= np.sum(pvalues)
        empty_choice = es[np.random.choice(range(len(es)), p=pvalues)]

        #i = np.random.randint(len(empty[0]))

        self.state[empty_choice[0], empty_choice[1]] = np.random.uniform(0, 1, 1)
        self.ages[empty_choice[0], empty_choice[1]] = 0

        return True

    def update_ages(self):
        self.ages[np.where(self.ages != -1)] += 1

    def update_state(self, moore=False):
        # Build a new state.
        new_state = deepcopy(self.state)

        # Find the current minimum value within the state.
        min_val = np.argmin(new_state)
        y = min_val // len(new_state[0])
        x = min_val % len(new_state[0])

        # Change the surrounding cells.
        if moore:
            combinations = itertools.product([-1, 0, 1], [-1, 0, 1])
        else:
            combinations = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        for xx, yy in combinations:
            xx = (xx + x) % len(new_state)
            yy = (yy + x) % len(new_state)

            if new_state[xx][yy] != 2:
                new_state[xx][yy] *= self.cell_decrease_factor

        # The cell with the minimum value moves to another grid, an empty cell is left.
        new_state[y][x] = 2
        self.ages[y][x] = -1

        # Save the state.
        self.state = new_state

        return True


def main():
    BaxSneppen2D()

if __name__ == '__main__':
    main()
