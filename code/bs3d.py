#
# Bax-Sneppen 2D implementation by
# Nicky, Maarten, Wessel and Willem
#

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import matplotlib.animation as animation


class BaxSneppen3D(object):
    def __init__(self, initial_values):
        self.states = [initial_values]
        self.ages = [np.zeros((len(initial_values), len(initial_values[0]), len(initial_values[0][0])))]

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
        z = min_val // (len(new_state[0]) * len(new_state[0][0]))
        y = min_val % (len(new_state[0]) * len(new_state[0][0])) // len(new_state[0][0])
        x = min_val % len(new_state[0][0])

        # Stopping criterium
        # if new_state[y][x] > 0.205:
        if new_state[z][y][x] > 0.10:
            return False

        # if len(self.states) > 50000:
        #     return False

        # Modify the values around the minimum value
        new_state[z][y][x] = np.random.uniform(0, 1, 1)
        new_state[z - 1][y][x] = np.random.uniform(0, 1, 1)
        new_state[z][y - 1][x] = np.random.uniform(0, 1, 1)
        new_state[z][y][x - 1] = np.random.uniform(0, 1, 1)
        new_state[(z + 1) % len(new_state)][y][x] = np.random.uniform(0, 1, 1)
        new_state[z][(y + 1) % len(new_state[0])][x] = np.random.uniform(0, 1, 1)
        new_state[z][y][(x + 1) % len(new_state[0][0])] = np.random.uniform(0, 1, 1)

        # Modify the cell ages
        new_ages[z][y][x] = 0
        new_ages[z - 1][y][x] = 0
        new_ages[z][y - 1][x] = 0
        new_ages[z][y][x - 1] = 0
        new_ages[(z + 1) % len(new_state)][y][x] = 0
        new_ages[z][(y + 1) % len(new_state[0])][x] = 0
        new_ages[z][y][(x + 1) % len(new_state[0][0])] = 0

        if moore:
            new_state[z - 1][y - 1][x - 1] = np.random.uniform(0, 1, 1)
            new_state[z - 1][y - 1][x] = np.random.uniform(0, 1, 1)
            new_state[z - 1][y - 1][(x + 1) % len(new_state[0][0])] = np.random.uniform(0, 1, 1)
            new_state[z - 1][y][x - 1] = np.random.uniform(0, 1, 1)
            new_state[z - 1][y][(x + 1) % len(new_state[0][0])] = np.random.uniform(0, 1, 1)
            new_state[z - 1][(y + 1) % len(new_state[0])][x - 1] = np.random.uniform(0, 1, 1)
            new_state[z - 1][(y + 1) % len(new_state[0])][x] = np.random.uniform(0, 1, 1)
            new_state[z - 1][(y + 1) % len(new_state[0])][(x + 1) % len(new_state[0][0])] = np.random.uniform(0, 1, 1)
            new_state[z][y - 1][x - 1] = np.random.uniform(0, 1, 1)
            new_state[z][(y + 1) % len(new_state[0])][x - 1] = np.random.uniform(0, 1, 1)
            new_state[z][y - 1][(x + 1) % len(new_state[0][0])] = np.random.uniform(0, 1, 1)
            new_state[z][(y + 1) % len(new_state[0])][(x + 1) % len(new_state[0][0])] = np.random.uniform(0, 1, 1)
            new_state[(z + 1) % len(new_state)][y - 1][x - 1] = np.random.uniform(0, 1, 1)
            new_state[(z + 1) % len(new_state)][y - 1][x] = np.random.uniform(0, 1, 1)
            new_state[(z + 1) % len(new_state)][y - 1][(x + 1) % len(new_state[0][0])] = np.random.uniform(0, 1, 1)
            new_state[(z + 1) % len(new_state)][y][x - 1] = np.random.uniform(0, 1, 1)
            new_state[(z + 1) % len(new_state)][y][(x + 1) % len(new_state[0][0])] = np.random.uniform(0, 1, 1)
            new_state[(z + 1) % len(new_state)][(y + 1) % len(new_state[0])][(x - 1) % len(new_state[0][0])] = np.random.uniform(0, 1, 1)
            new_state[(z + 1) % len(new_state)][(y + 1) % len(new_state[0])][x] = np.random.uniform(0, 1, 1)
            new_state[(z + 1) % len(new_state)][(y + 1) % len(new_state[0])][(x + 1) % len(new_state[0][0])] = np.random.uniform(0, 1, 1)


            new_ages[z - 1][y - 1][x - 1] = 0
            new_ages[z - 1][y - 1][x] = 0
            new_ages[z - 1][y - 1][(x + 1) % len(new_state[0][0])] = 0
            new_ages[z - 1][y][x - 1] = 0
            new_ages[z - 1][y][(x + 1) % len(new_state[0][0])] = 0
            new_ages[z - 1][(y + 1) % len(new_state[0])][x - 1] = 0
            new_ages[z - 1][(y + 1) % len(new_state[0])][x] = 0
            new_ages[z - 1][(y + 1) % len(new_state[0])][(x + 1) % len(new_state[0][0])] = 0
            new_ages[z][y - 1][x - 1] = 0
            new_ages[z][(y + 1) % len(new_state[0])][x - 1] = 0
            new_ages[z][y - 1][(x + 1) % len(new_state[0][0])] = 0
            new_ages[z][(y + 1) % len(new_state[0])][(x + 1) % len(new_state[0][0])] = 0
            new_ages[(z + 1) % len(new_state)][y - 1][x - 1] = 0
            new_ages[(z + 1) % len(new_state)][y - 1][x] = 0
            new_ages[(z + 1) % len(new_state)][y - 1][(x + 1) % len(new_state[0][0])] = 0
            new_ages[(z + 1) % len(new_state)][y][x - 1] = 0
            new_ages[(z + 1) % len(new_state)][y][(x + 1) % len(new_state[0][0])] = 0
            new_ages[(z + 1) % len(new_state)][(y + 1) % len(new_state[0])][(x - 1) % len(new_state[0][0])] = 0
            new_ages[(z + 1) % len(new_state)][(y + 1) % len(new_state[0])][x] = 0
            new_ages[(z + 1) % len(new_state)][(y + 1) % len(new_state[0])][(x + 1) % len(new_state[0][0])] = 0

        #
        self.states.append(new_state)
        self.ages.append(new_ages)

        return True

    def plot_ages(self):
        plt.imshow(self.ages[-1], aspect='auto', cmap='jet_r', interpolation='nearest')


def main():
    initial_values = np.random.rand(10, 10, 10)
    bs3d = BaxSneppen3D(initial_values)
    bs3d.execute(True)
    bs3d.plot_ages()
    animate_ages(bs3d.ages)


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
