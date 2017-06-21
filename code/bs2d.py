#
# Bax-Sneppen 2D implementation by
# Nicky, Maarten, Wessel and Willem
#
from copy import deepcopy
from scipy.stats import mvn
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import multivariate_normal


class BaxSneppen2D(object):
    def __init__(self, slum_size=(15, 15), empty_percent=0.3):
        self.state = np.ones(slum_size) * 2
        self.ages = np.ones(slum_size) * -1
        self.populate(empty_percent, slum_size)
        self.avalanches = []
        self.cur_av_count = 0
        self.cur_av_start = -1
        self.size = slum_size
        xmean = self.size[0]*0.5
        ymean = self.size[1]*0.5
        cov = np.array([[xmean*0.2, 0], [0, ymean*0.2]])
        self.mvn = multivariate_normal([xmean, ymean], cov)

    def populate(self, empty_percent, slum_size):
        for x in range(slum_size[0]):
            for y in range(slum_size[1]):
                self.state[y][x] = np.random.uniform(0, 1, 1)
                self.ages[y][x] = 0

        empty = np.random.choice(range(len(self.state[0]) * len(self.state)),
                                 empty_percent * slum_size[0] * slum_size[1])

        for i in empty:
            self.state[i % slum_size[0]][i // slum_size[0]] = 2
            self.ages[i % slum_size[0]][i // slum_size[0]] = -1

        # r = math.sqrt(slum_size[0]*slum_size[1]*(1-empty_percent)/math.pi)
        # mx = slum_size[0]*0.5
        # my = slum_size[1]*0.5
        # for x in range(slum_size[0]):
        #     for y in range(slum_size[1]):
        #         if (y-my)**2 + (x-mx)**2 < r**2:
        #             self.state[y][x] = np.random.uniform(0, 1, 1)
        #             self.ages[y][x] = 0

    def execute(self, moore=False):
        while self.update_state(moore):
            continue

        print(self.ages)
        print(len(self.state))

    def get_min_val(self):
        return np.min(self.state)

    def get_min_val_index(self):
        return np.argmin(self.state)

    def get_avg_val(self):
        return np.average([i for i in self.state.flatten() if i != 2])

    def full_cells(self):
        return len(np.where(self.state < 2)[0])

    def has_empty(self):
        empty = np.where(self.state == 2)

        if len(empty[0]) == 0:
            return False

        return True

    def add_to_grid(self, original_value):
        # TODO use original_value in a useful manner
        original_value = original_value
        empty = np.where(self.state == 2)
        if len(empty[0]) == 0:
            return False


        es = [(x,y) for x,y in zip(empty[0], empty[1])]
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
        # Build a new state
        new_state = deepcopy(self.state)

        min_val = np.argmin(new_state)
        y = min_val // len(new_state[0])
        x = min_val % len(new_state[0])
        # Stopping criterium
        # if new_state[y][x] > 0.205:
        # if new_state[y][x] > 0.20:
        #     return False

        # Count avalanches
        if new_state[y][x] > self.cur_av_start:
            if self.cur_av_start != -1:
                self.avalanches.append(self.cur_av_count)
            self.cur_av_count = 0
            self.cur_av_start = new_state[y][x]
        else:
            self.cur_av_count += 1

        # if len(self.ages) > 50000:
        #     return False

        if moore:
            for xx, yy in itertools.product([-1, 0, 1], [-1, 0, 1]):
                # Modify the values around the minimum value
                if new_state[(y + yy) % len(new_state)][(x + xx) % len(new_state)] != 2:
                    new_state[(y + yy) % len(new_state)][(x + xx) % len(new_state)] = \
                        np.random.uniform(0, 1, 1)
                    # Modify the cell ages
                    # self.ages[(y + yy) % len(new_state)][(x + xx) % len(new_state)] = 0
        else:
            for xx, yy in [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]]:
                # Modify the values around the minimum value
                if new_state[(y + yy) % len(new_state)][(x + xx) % len(new_state)] != 2:
                    new_state[(y + yy) % len(new_state)][(x + xx) % len(new_state)] = \
                        np.random.uniform(0, 1, 1)
                    # Modify the cell ages
                    # self.ages[(y + yy) % len(new_state)][(x + xx) % len(new_state)] = 0

        # Wait, the person who left, left an empty house
        new_state[y][x] = 2
        self.ages[y][x] = -1
        self.state = new_state

        return True

    def plot_ages(self):
        plt.imshow(self.ages[-1], aspect='auto', cmap='jet_r', interpolation='nearest')

    def plot_avalanches(self):
        plt.plot(self.avalanches)
        plt.show()

        plt.show()


def main():
    bs2d = BaxSneppen2D()
    bs2d.execute(False)
    bs2d.plot_avalanches()
    # bs2d.plot_ages()
    # animate_ages(bs2d.ages)


def animate_ages(ages):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(ages[1], aspect='auto', cmap='jet_r', interpolation='nearest',
                   vmin=0, vmax=np.max(ages[-1]))

    def animate(i):
        im.set_array(ages[i])  # update the data
        fig.canvas.draw()
        plt.title('iteration: ' + str(i))
        return im

    animation.FuncAnimation(fig, animate, range(int(len(ages) * 0.75), len(ages)),
                            interval=2, blit=False)
    plt.show()


if __name__ == '__main__':
    main()
