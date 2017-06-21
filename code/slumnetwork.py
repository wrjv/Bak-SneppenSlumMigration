from bs2d import BaxSneppen2D
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import matplotlib.animation as animation
from math import *
import scipy.stats


class Slums(object):

    def __init__(self, n_slums, slum_size=(15, 15), empty_percent=0.75):
        self.slum_list = [BaxSneppen2D(slum_size, empty_percent) for _ in range(n_slums)]
        self.total_cells = slum_size[0] * slum_size[1] * n_slums
        self.empty_percent = empty_percent        
        self.states = []
        self.time = 0

    def execute(self, moore=False):
        while self.update_state(moore):
            self.states.append(deepcopy(self.slum_list))
            continue
        self.plot_slums()

    def update_state(self, moore=False):
        min_vals = [slum.get_min_val() for slum in self.slum_list]

        for slum in self.slum_list:
            slum.update_ages()

        self.slum_list[np.argmin(min_vals)].update_state(moore)

        to_slum = self.find_optimal_location(np.argmin(min_vals))

        self.slum_list[to_slum].add_to_grid(min(min_vals))

        if self.time > 3000:
            return False

        self.time += 1
        return True

    def find_optimal_location(self, origin_slum):
        parameters = []

        slot_distrib = scipy.stats.norm(self.total_cells * (1 - self.empty_percent), self.total_cells * self.empty_percent * 0.5)

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

            pvalues.append(parameters[i][0] * slot_distrib.pdf(parameters[i][2]))

        pvalues = pvalues / sum(pvalues)

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

        ani = animation.FuncAnimation(f, animate, range(int(len(self.states) * 0.3), len(self.states)), interval=2, blit=False)
        plt.show()




def main():
    slums = Slums(4, (20, 20))
    slums.execute()

if __name__ == '__main__':
    main()
