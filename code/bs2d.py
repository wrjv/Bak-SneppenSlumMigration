#
# Bax-Sneppen 2D implementation by
# Nicky, Maarten, Wessel and Willem
#

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import matplotlib.animation as animation
import itertools


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
        # if new_state[y][x] > 0.205:
        # if new_state[y][x] > 0.20:
        #     return False

        if len(self.states) > 50000:
            return False

        if moore:
            for xx,yy in itertools.product([-1,0,1],[-1,0,1]):
                # Modify the values around the minimum value
                new_state[(y + yy) % len(new_state)][(x + xx) % len(new_state)] = np.random.uniform(0, 1, 1)
                # Modify the cell ages
                new_ages[(y + yy) % len(new_state)][(x + xx) % len(new_state)] = 0
        else:
                new_ages[y][x] = 0
                new_state[y][x] = np.random.uniform(0, 1, 1)
                for xx,yy in [[-1,0], [1,0], [0,-1], [0,1]]:
                    # Modify the values around the minimum value
                    new_state[(y + yy) % len(new_state)][(x + xx) % len(new_state)] = np.random.uniform(0, 1, 1)
                    # Modify the cell ages
                    new_ages[(y + yy) % len(new_state)][(x + xx) % len(new_state)] = 0

        self.states.append(new_state)
        self.ages.append(new_ages)

        return True

    def plot_ages(self):
        plt.imshow(self.ages[-1], aspect='auto', cmap='jet_r', interpolation='nearest')


def main():
    initial_values = np.random.rand(100, 100)
    bs2d = BaxSneppen2D(initial_values)
    bs2d.execute(True)
    bs2d.plot_ages()
    animate_ages(bs2d.ages)


def animate_ages(ages):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(ages[1], aspect='auto', cmap='jet_r', interpolation='nearest', vmin=0, vmax=np.max(ages[-1]))

    def animate(i):
        im.set_array(ages[i])  # update the data
        fig.canvas.draw()
        plt.title('iteration: ' + str(i))
        return im

    ani = animation.FuncAnimation(fig, animate, range(int(len(ages) * 0.75), len(ages)), interval=2, blit=False)
    plt.show()


if __name__ == '__main__':
    main()
