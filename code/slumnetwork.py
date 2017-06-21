from bs2d import BaxSneppen2D
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import matplotlib.animation as animation
from math import *

class Slums(object):

    def __init__(self, n_slums, slum_size=(15, 15), empty_percent=0.25):
        self.slum_list = [BaxSneppen2D(slum_size, empty_percent) for _ in range(n_slums)]
        self.states = []
        self.time = 0

    def execute(self, moore=False):
        while self.update_state(moore):
            self.states.append(deepcopy(self.slum_list))
            continue

    def update_state(self, moore=False):
        min_vals = [slum.get_min_val() for slum in self.slum_list]

        for slum in self.slum_list:
            slum.update_ages()

        self.slum_list[np.argmin(min_vals)].update_state(moore)

        to_slum = self.find_optimal_location(np.argmin(min_vals))

        self.slum_list[to_slum].add_to_grid(min(min_vals))

        if self.time > 10000:
            return False

        self.time += 1
        return True

    def find_optimal_location(self, origin_slum):
        # Te vol is niet fijn
        origin_avg = self.slum_list[origin_slum].get_avg_val()

        parameters = {}

        # De average satisfaction moet het liefst hoger zijn!
        for i in range(len(self.slum_list)):
            parameters[i] = [self.slum_list[i].get_avg_val(),
                             self.slum_list[i].has_empty()]

        total_fitness = sum([slum[0] if slum[1] else 0 for slum in parameters.values()])

        pvalues = []

        for i in range(len(self.slum_list)):
            if parameters[i][1]:
                parameters[i][0] /= total_fitness
            else:
                parameters[i][0] = 0

            pvalues.append(parameters[i][0])

        return np.random.choice(range(len(self.slum_list)), 1, p=pvalues)

    def plot_slums(self):
        cols = ceil(len(self.slum_list)**0.5)
        rows = ceil(len(self.slum_list)/cols)

        f, axarr = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True)

        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.xticks([]); plt.yticks([])

        ims = list()
        max_ages = [np.max(slum.ages) for slum in self.slum_list]
        max_age = max(max_ages)

        cmap = plt.cm.jet_r
        cmap.set_under((1, 1, 1, 1))

        if len(self.slum_list) > 1:
            for slum, ax in zip(self.states[0], axarr.flatten()):
                ims.append(ax.imshow(slum.ages, aspect='auto', cmap=cmap, interpolation='nearest', vmin=0, vmax=max_age))
        elif len(self.slum_list) == 1:
            for slum in self.states[0]:
                ims.append(axarr.imshow(slum.ages, aspect='auto', cmap=cmap, interpolation='nearest', vmin=0, vmax=max_age))
        else:
            assert False

        def animate(i):
            plt.suptitle('iteration: ' + str(i))
            for slum, im, in zip(self.states[i], ims):
                im.set_array(slum.ages)
            f.canvas.draw()
            return ims

        ani = animation.FuncAnimation(f, animate, range(int(len(self.states) * 0), len(self.states)), interval=2, blit=False)
        plt.show()


    def plot_barrier_distribution(self):
        barriers = []
        for slum in self.states[-1]:
            barriers = barriers + list(slum.state[np.where(slum.state <= 1)].flatten())
        print(barriers)
        plt.hist(barriers, bins=30, range=(0,1))
        plt.show()


def main():
    slums = Slums(4, (50,50))
    slums.execute()

if __name__ == '__main__':
    main()
