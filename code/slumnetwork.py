from math import ceil
from copy import deepcopy
from bs2d import BaxSneppen2D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.stats


class Slums(object):

    def __init__(self, n_slums, slum_size=(15, 15), empty_percent=0.25, random_select=False,
                 time_limit=10000):
        self.slum_list = [BaxSneppen2D(slum_size, empty_percent) for _ in range(n_slums)]
        assert slum_size[0] == slum_size[1]
        self.slum_size = slum_size[0]
        self.total_cells = slum_size[0] * slum_size[1] * n_slums
        self.empty_percent = empty_percent
        self.random_select = random_select
        self.states = []
        self.time = 0
        self.previous_location = [(0,0) for _ in range(n_slums)]
        self.distances = [[] for _ in range(n_slums)]
        self.avalanche_size = [0]
        self.aval_start_val = 0
        self.time_limit = time_limit

    def execute(self, moore=False, save_steps=25):
        iterator = 0
        self.save_steps = save_steps
        while self.update_state(moore):
            if iterator % save_steps == 0:
                self.states.append(deepcopy(self.slum_list))
            iterator += 1
            continue

    def update_state(self, moore=False):
        min_vals = [slum.get_min_val() for slum in self.slum_list]

        # calculate the distance between mutations
        min_vals_indexes = [slum.get_min_val_index() for slum in self.slum_list]
        x, y = min_vals_indexes[np.argmin(min_vals)]//self.slum_size, min_vals_indexes[np.argmin(min_vals)]%self.slum_size
        xold, yold = self.previous_location[np.argmin(min_vals)]
        distance = ((x-xold)**2 + (y-yold)**2)**0.5
        self.distances[np.argmin(min_vals)].append(distance)
        self.previous_location[np.argmin(min_vals)] = (x,y)

        # calculate the size of the avalanche
        if min(min_vals) >= self.aval_start_val:
            self.avalanche_size.append(1)
            self.aval_start_val = min(min_vals)
        else:
            self.avalanche_size[-1] += 1

        for slum in self.slum_list:
            slum.update_ages()

        self.slum_list[np.argmin(min_vals)].update_state(moore)

        if self.random_select:
            to_slum = self.alt_find_optimal_location(np.argmin(min_vals))
        else:
            to_slum = self.find_optimal_location(np.argmin(min_vals))

        self.slum_list[to_slum].add_to_grid(min(min_vals))

        if self.time > self.time_limit:
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
                parameters[i].append((parameters[i][1] - parameters[origin_slum][1] + 1)**2)
            else:
                parameters[i].append(0)

            pvalues.append(parameters[i][3])

        pvalues = np.array(pvalues) / sum(pvalues)

        return np.random.choice(range(len(self.slum_list)), 1, p=pvalues)

    def alt_find_optimal_location(self, origin_slum):
        parameters = []

        for i in range(len(self.slum_list)):
            parameters.append(self.slum_list[i].has_empty())

        has_any_empty = sum([1 if has_empty else 0 for has_empty in parameters])

        if has_any_empty == 0:
            return origin_slum

        pvalues = []

        for i in range(len(self.slum_list)):
            if parameters[i]:
                pvalues.append(1)
            else:
                pvalues.append(0)

        pvalues = pvalues / sum(pvalues)

        return np.random.choice(range(len(self.slum_list)), 1, p=pvalues)

    def plot_slums(self, start):
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
            plt.suptitle('iteration: ' + str(i*self.save_steps))
            for slum, im, in zip(self.states[i], ims):
                im.set_array(slum.ages)
            f.canvas.draw()
            return ims

        ani = animation.FuncAnimation(f, animate, range(int(len(self.states) * start),
                                                        len(self.states), 1), interval=2, blit=False)
        plt.show()


    def plot_barrier_distribution(self):
        barriers = []
        minima = []
        for slum in self.slum_list:
            barriers = barriers + list(slum.state[np.where(slum.state <= 1)].flatten())
        for timestep in self.states:
            minima.append(min([state.get_min_val() for state in timestep]))

        (counts_min, bins_min, _) = plt.hist(minima, bins=30)
        (counts_bar, bins_bar, _) = plt.hist(barriers, bins=30)
        plt.clf()
        plt.plot(bins_min[:-1], counts_min/sum(counts_min), label="minimum barriers")
        plt.plot(bins_bar[:-1], counts_bar/sum(counts_bar), label="barrier distribution")
        plt.title("barrier and minumum barriers distribution")
        plt.legend()
        plt.xlabel("B")
        plt.ylabel("P(B)")
        plt.show()

    def plot_avalanche_distance(self):
        avalanches = []
        for each in self.distances:
            avalanches = avalanches + each
            # print(avalanches)

        (counts, bins, _) = plt.hist(avalanches, bins=30)
        plt.clf()
        plt.loglog(bins[:-1], counts/sum(counts))
        plt.title("distance between succesive mutations")
        plt.xlabel(r"$log_{10}(X)$")
        plt.ylabel(r"$log_{10}(C(X))$")
        plt.show()

    def plot_avalanche_size(self):
        (counts, bins, _) = plt.hist(self.avalanche_size, bins=30)
        plt.clf()
        plt.loglog(bins[:-1], counts/sum(counts))
        plt.show()

def main():
    slums = Slums(4, (30, 30), empty_percent=0.06, time_limit=15000)
    slums.execute(save_steps=50)
    slums.plot_barrier_distribution()
    slums.plot_avalanche_distance()
    slums.plot_avalanche_size()
    #slums.plot_slums(start=0)

if __name__ == '__main__':
    main()
