from math import ceil
from copy import deepcopy
from bs2d import BaxSneppen2D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.stats


class Slums(object):

    def __init__(self, n_slums, slum_size=(15, 15), empty_percent=0.25, random_select=False):
        self.slum_list = [BaxSneppen2D(slum_size, empty_percent) for _ in range(n_slums)]
        self.total_cells = slum_size[0] * slum_size[1] * n_slums
        self.empty_percent = empty_percent
        self.random_select = random_select
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

        if self.random_select:
            to_slum = self.alt_find_optimal_location(np.argmin(min_vals))
        else:
            to_slum = self.find_optimal_location(np.argmin(min_vals))

        self.slum_list[to_slum].add_to_grid(min(min_vals))

        if self.time > 10000:
            return False

        self.time += 1
        return True

    def find_optimal_location(self, origin_slum):
        parameters = []

        # De average satisfaction moet het liefst zo hoog mogelijk zijn!
        for i in range(len(self.slum_list)):
            parameters.append([self.slum_list[i].get_avg_val(),
                               self.slum_list[i].has_empty(),
                               self.slum_list[i].full_cells()])

        total_fitness = sum([slum[0] if slum[1] else 0 for slum in parameters])

        if total_fitness == 0:
            return origin_slum

        pvalues = []

        for i in range(len(self.slum_list)):
            if parameters[i][1]:
                parameters[i][0] /= total_fitness
            else:
                parameters[i][0] = 0

            pvalues.append(parameters[i][0])# * slot_distrib.pdf(parameters[i][2]))

        pvalues = pvalues / sum(pvalues)

        return np.random.choice(range(len(self.slum_list)), 1, p=pvalues)


    def alt_find_optimal_location(self, origin_slum):
        parameters = []

        for i in range(len(self.slum_list)):
            parameters.append(self.slum_list[i].has_empty())

        has_any_empty = sum([1 if has_empty else 0 for has_empty in parameters])

        if has_any_empty == 0:
            return origin_slum

        for i in range(len(self.slum_list)):
            if parameters[i]:
                pvalues.append(1)
            else:
                pvalues.append(0)

        pvalues = pvalues / sum(pvalues)

        return np.random.choice(range(len(self.slum_list)), 1, p=pvalues)

    def plot_slums(self, start, show_steps):
        cols = ceil(len(self.slum_list)**0.5)
        rows = ceil(len(self.slum_list)/cols)

        f, axarr = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True)

        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.xticks([])
        plt.yticks([])

        ims = list()
        max_ages = [np.max(slum.ages) for slum in self.slum_list]
        max_age = max(max_ages)

        cmap = plt.cm.jet_r
        cmap.set_under((1, 1, 1, 1))

        if len(self.slum_list) > 1:
            for slum, ax in zip(self.states[0], axarr.flatten()):
                ims.append(ax.imshow(slum.ages, aspect='auto', cmap=cmap, interpolation='nearest',
                                     vmin=0, vmax=max_age))
        elif len(self.slum_list) == 1:
            for slum in self.states[0]:
                ims.append(axarr.imshow(slum.ages, aspect='auto', cmap=cmap,
                                        interpolation='nearest', vmin=0, vmax=max_age))
        else:
            assert False

        def animate(i):
            plt.suptitle('iteration: ' + str(i))
            for slum, im, in zip(self.states[i], ims):
                im.set_array(slum.ages)
            f.canvas.draw()
            return ims

        ani = animation.FuncAnimation(f, animate, range(int(len(self.states) * start),
                                                        len(self.states), show_steps), interval=2, blit=False)
        plt.show()


    def plot_barrier_distribution(self):
        barriers = []
        for slum in self.states[-1]:
            barriers = barriers + list(slum.state[np.where(slum.state <= 1)].flatten())
        plt.hist(barriers, bins=30, range=(0, 1))
        plt.show()


def main():
    slums = Slums(9, (25, 25), empty_percent=0.06)
    slums.execute()
    slums.plot_slums(start=0, show_steps=25)

if __name__ == '__main__':
    main()
