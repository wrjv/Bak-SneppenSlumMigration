'''
A slum network implementation based on 2D Bax-Sneppen models joined together in a network.

This code was built by Nicky Kessels, Maarten van der Sande, Wessel Klijnsma and Willem Vermeulen,
all master students Computational Science at the University of Amsterdam.
'''

from math import ceil
from copy import deepcopy
from bs2d import BaxSneppen2D
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import gaussian_kde


class Slums(object):
    '''
    Encapsules a slum simulation based on 2D Bax-Sneppen models joined together in a network.
    '''

    # pylint: disable=too-many-instance-attributes, too-many-arguments

    # It's reasonable to have 12 variables, to be able to keep track of all parameters
    # and all slums.

    def __init__(self, n_slums, slum_size=(15, 15), empty_percent=0.25, random_select=False,
                 time_limit=10000):
        # Set some overall parameters.
        self.time = 0
        self.time_limit = time_limit

        self.save_steps = 1

        self.slum_size = slum_size[0]

        self.random_select = random_select

        self.threshold = 0

        self.avalanche_size = [0]
        self.avalanche_sizes = []
        self.aval_start_val = 0

        # Set some variables to keep track of all slums.
        self.slum_list = [BaxSneppen2D(slum_size, empty_percent) for _ in range(n_slums)]
        self.states = []

        self.previous_location = [(0, 0) for _ in range(n_slums)]
        self.distances = [[] for _ in range(n_slums)]
        self.barrier_dists = []

    def execute(self, moore=False, save_steps=25):
        '''
        Executes the slum simulation.

        PARAMETERS
        ===================================================
        moore: boolean
        Wether a moore neighbourhood should be used or not.

        save_steps: integer
        The number of steps between each state saved, used
        in further animations.
        '''

        iterator = 0

        # Make sure the animation function knows the distance between
        # each step.
        self.save_steps = save_steps

        # Keep updating the states until the time limit is reached.
        while self.update_state(moore):
            if iterator % save_steps == 0:
                self.states.append(deepcopy(self.slum_list))
                (counts, bins, _) = plt.hist(self.avalanche_size, bins=200)
                self.avalanche_sizes.append((counts, bins))
                self.barrier_dists.append(self.get_barrier_distribution())

            iterator += 1

    def update_state(self, moore=False):
        '''
        Updates the current state. Used by the execution
        function.

        PARAMETERS
        ===================================================
        moore: boolean
        Whether a moore neighbourhood should be used or not.

        RETURNS
        ===================================================
        boolean
        Whether the time limit has been reached or not.
        '''

        min_vals = [slum.get_min_val() for slum in self.slum_list]
        min_slum = np.argmin(min_vals)

        # Find the last location and new location
        xold, yold = self.previous_location[min_slum]

        min_vals_indexes = [slum.get_min_val_index() for slum in self.slum_list]

        xnew, ynew = min_vals_indexes[min_slum] // self.slum_size, min_vals_indexes[
            np.argmin(min_vals)] % self.slum_size

        # Calculate the distances between both mutations
        distance = ((xnew - xold) ** 2 + (ynew - yold) ** 2) ** 0.5
        self.distances[min_slum].append(distance)
        self.previous_location[min_slum] = (xnew, ynew)

        # Start a new avalanche or update the size of a current one.
        if min(min_vals) >= self.aval_start_val:
            self.avalanche_size.append(1)
            self.aval_start_val = min(min_vals)
        else:
            self.avalanche_size[-1] += 1

        # Update the ages of all cells in all slums.
        for slum in self.slum_list:
            slum.update_ages()

        # Update the state of the slum containing the minimum value.
        self.slum_list[min_slum].update_state(moore)

        # Determine to what other slum the cell goes.
        to_slum = self.get_to_slum(min_slum)

        # Add another new slum with a small chance.
        # print(np.mean(self.slum_list[min_slum].state[self.slum_list[min_slum].state != 2]))
        if np.random.uniform(0, 1, 1) < self.threshold - np.mean(
                self.slum_list[min_slum].state[self.slum_list[min_slum].state != 2]):
            print("New slum built.")
            self.slum_list.append(BaxSneppen2D((self.slum_size, self.slum_size), empty_percent=1))
            self.previous_location.append((0, 0))
            self.distances.append([])
            to_slum = -1

        # Add new people to the grid.
        self.slum_list[to_slum].add_to_grid(min(min_vals))
        # to_slum = self.get_to_slum(min_vals)
        # self.slum_list[to_slum].add_to_grid()

        # Check if the time limit is reached, otherwise return False (and
        # the execution ends)
        if self.time > self.time_limit:
            return False

        self.time += 1
        return True

    def get_to_slum(self, min_slum):
        '''
        Determines the next slum a cell wants to go to.

        PARAMETERS
        ===================================================
        min_slum: integer
        The index of the slum with the current minimum
        value within the system.

        RETURNS
        ===================================================
        integer
        The index of the slum assigned to the cell.
        '''

        if self.random_select:
            return self.alt_find_optimal_location()

        return self.find_optimal_location(min_slum)

    def find_optimal_location(self, origin_slum):
        '''
        Determines the next slum a cell wants to go to,
        with a preference to cells with cells that are more
        satisfied.

        PARAMETERS
        ===================================================
        origin_slum: integer
        The index of the slum with the current minimum
        value within the system.

        RETURNS
        ===================================================
        integer
        The index of the slum assigned to the cell.
        '''

        # Gather all parameters needed for calculating a p-value for each slum.
        parameters = []

        for i in range(len(self.slum_list)):
            parameters.append([self.slum_list[i].get_avg_val(),
                               self.slum_list[i].has_empty()])

        # Calculate a p-value for each slum.
        pvalues = []

        for i in range(len(self.slum_list)):
            # Check if there is room for a new cell in the grid. Assign a non-normalised p-value
            # to the slum. Slums with a lower satisfaction than the original slum are less likely
            # to be chosen by a cell.
            if parameters[i][1]:
                parameters[i].append((parameters[i][1] - parameters[origin_slum][1] + 1) ** 2)
            else:
                parameters[i].append(0)

            pvalues.append(parameters[i][2])

        # Normalise the pvalues and make a choice of a location for a cell to go to.
        pvalues = np.array(pvalues) / sum(pvalues)

        return np.random.choice(range(len(self.slum_list)), 1, p=pvalues)

    def alt_find_optimal_location(self):
        '''
        Determines the next slum a cell wants to go to,
        randomised.

        PARAMETERS
        ===================================================
        None

        RETURNS
        ===================================================
        integer
        The index of the slum assigned to the cell.
        '''

        # Gather information on each slum whether it has room for a new cell or not.
        parameters = []

        for i in range(len(self.slum_list)):
            parameters.append(self.slum_list[i].has_empty())

        # Assign an equal probability to each slum.
        pvalues = []

        for i in range(len(self.slum_list)):
            # Check if there is room for a new cell in the grid.
            if parameters[i]:
                pvalues.append(1)
            else:
                pvalues.append(0)

        # Normalise the pvalues and make a choice of a location for a cell to go to.
        pvalues = pvalues / sum(pvalues)

        return np.random.choice(range(len(self.slum_list)), 1, p=pvalues)

    def plot_slums(self, start):
        global ns
        cols = ceil(len(self.slum_list) ** 0.5)
        rows = ceil(len(self.slum_list) / cols)

        f, axarr = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True)

        # remove the axes labels and set spacing
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.xticks([])
        plt.yticks([])

        # set the colour map
        cmap = self.get_colormap()

        # calculate the max age for the plot
        max_age = max([np.max(slum.ages) for slum in self.slum_list])

        # initialize the plot
        ims = list()

        if len(self.slum_list) == 1:
            axarr = np.array([axarr])

        ns = len(self.states[0])
        for slum, ax in zip(self.states[0], axarr.flatten()):
            ims.append(ax.imshow(slum.ages, aspect='auto', cmap=cmap, interpolation='nearest',
                                 vmin=0, vmax=max_age))

        # animate
        def animate(i):
            global ns
            if len(self.states[i]) > ns:
                for slum, ax in zip(self.states[i][ns:len(self.states[i])],
                                    axarr.flatten()[ns:len(self.states[i])]):
                    ims.append(ax.imshow(slum.ages, aspect='auto', cmap=cmap,
                                         interpolation='nearest', vmin=0, vmax=max_age))
                ns = len(self.states[i])

            plt.suptitle('iteration: ' + str(i * self.save_steps))
            for slum, im, in zip(self.states[i], ims):
                im.set_array(slum.ages)
            f.canvas.draw()
            if i == len(self.states) - 1:
                for im in ims:
                    im.set_array(np.ones((self.slum_size, self.slum_size)) * -1)

            return ims

        _ = animation.FuncAnimation(f, animate, range(int(len(self.states) * start),
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
        plt.plot(bins_min[:-1], counts_min / sum(counts_min), label="minimum barriers")
        plt.plot(bins_bar[:-1], counts_bar / sum(counts_bar), label="barrier distribution")
        plt.title("barrier and minumum barriers distribution")
        plt.legend()
        plt.xlabel("B")
        plt.ylabel("P(B)")
        plt.show()

    def get_barrier_distribution(self):
        barriers = []
        minima = []
        for slum in self.slum_list:
            barriers = barriers + list(slum.state[np.where(slum.state <= 1)].flatten())
        for timestep in self.states:
            minima.append(min([state.get_min_val() for state in timestep]))

        # (counts_min, bins_min, _) = plt.hist(minima, bins=30)
        # (counts_bar, bins_bar, _) = plt.hist(barriers, bins=30)
        if len(minima) == 1:
            minima.append(0)

        minima = np.array(minima)
        barriers = np.array(barriers)
        min_density = gaussian_kde(minima)
        bar_density = gaussian_kde(barriers)
        return bar_density, min_density

    def plot_avalanche_distance(self):
        avalanches = []
        for each in self.distances:
            avalanches = avalanches + each
            # print(avalanches)

        (counts, bins, _) = plt.hist(avalanches, bins=30)
        plt.clf()
        plt.loglog(bins[:-1], counts / sum(counts))
        plt.title("distance between succesive mutations")
        plt.xlabel(r"$log_{10}(X)$")
        plt.ylabel(r"$log_{10}(C(X))$")
        plt.show()

    def plot_avalanche_size(self):
        (counts, bins, _) = plt.hist(self.avalanche_size, bins=30)
        plt.clf()
        plt.loglog(bins[:-1], counts / sum(counts))
        plt.title("avalanche sizes")
        plt.xlabel(r"S$")
        plt.ylabel(r"P(S)$")
        plt.show()

    def plot_growth_over_time(self):
        growths = [[] for _ in range(len(self.states[-1]))]
        scaler = self.save_steps
        for state in self.states:
            for index in range(len(state)):
                growths[index].append(state[index].full_cells())
        for slum in growths:
            maxim = len(growths[0])
            minim = len(slum)
            plt.plot(range((maxim - minim) * scaler, maxim * scaler, scaler), slum)
        plt.title("growth of slums over time")
        plt.xlabel("number of iterations")
        plt.ylabel("population size of slum")
        plt.show()

    def get_colormap(self):
        cmap = plt.cm.jet_r
        cmap.set_under((1, 1, 1, 1))
        return cmap

    def setup_slum_anim(self):
        # TODO Maarten: 2, 5 & 6 plots dont seem to work?

        slumaxarr = []
        size = len(self.states[-1])
        cols = ceil(size ** 0.5)
        rows = ceil(size / cols)

        f = plt.figure()
        for i in range(size):
            slumaxarr.append(plt.subplot2grid((1 + rows, 1 + cols), (i // rows, (i % rows) % cols)))

        for i, slumax in enumerate(slumaxarr):
            # plt.subplots_adjust(wspace=0.05, hspace=0.05)
            slumax.imshow(self.states[-1][i].ages)
            slumax.set_xticklabels([])
            slumax.set_yticklabels([])

        ims = list()
        ns = len(self.states[0])
        for slum, ax in zip(self.states[0], slumaxarr):
            ims.append(ax.imshow(slum.ages, aspect='auto', cmap=cmap, interpolation='nearest',
                                 vmin=0, vmax=max_age))
        return f, ims, rows, cols, ns, slumaxarr

    def make_dashboard(self):
        global coolax, ns, cmap, max_age

        cmap = self.get_colormap()
        max_age = max([np.max(slum.ages) for slum in self.slum_list])

        f, ims, rows, cols, ns, slumaxarr = self.setup_slum_anim()

        # TODO store parameters in such a way that we have the history of them
        coolax = plt.subplot2grid((1 + rows, cols + 1), (rows, 0))

        if len(self.avalanche_sizes[-1][1]) > len(self.avalanche_sizes[-1][0]):
            xs = self.avalanche_sizes[-1][1][:-1]
            xs_original = self.avalanche_sizes[-1][1][:-1]
        ys = self.avalanche_sizes[-1][0] / sum(self.avalanche_sizes[-1][0])

        bin_size = xs[1] / 2.0

        pairs = [pair for pair in zip(xs, ys) if pair[1] != 0]
        xs = [pair[0] + bin_size for pair in pairs]
        ys = [pair[1] for pair in pairs]

        line, = coolax.loglog(xs, ys, ".")

        popt, pcov = curve_fit(powerlaw, xs, ys)

        line_fit, = plt.plot(xs_original, powerlaw(xs_original, *popt), 'r-', label=r'$K=' + str(np.round(popt[1], 3)) + "$")

        plt.title("avalanche sizes")
        plt.legend()
        plt.xlabel(r"$log_{10}(S)$")
        plt.ylabel(r"$log_{10}(P(S))$")

        # min_x = self.barrier_dists[-1][1][1]
        # min_y = self.barrier_dists[-1][1][0] / sum(self.barrier_dists[-1][1][0])
        bdax = plt.subplot2grid((1 + rows, cols + 1), (rows, 1))
        # line_min, = bdax.plot(self.barrier_dists[-1][0][1][:-1], self.barrier_dists[-1][0][0] / sum(self.barrier_dists[-1][0][0]), label='minima')
        # line_bd, = bdax.plot(self.barrier_dists[-1][1][1][:-1],
        #                    self.barrier_dists[-1][1][0] / sum(self.barrier_dists[-1][1][0]), label='barriers')
        xs_space = np.linspace(0, 1, 300)
        line_min, = bdax.plot(xs, self.barrier_dists[-1][1](xs), label='minima')
        line_bd, = bdax.plot(xs, self.barrier_dists[-1][0](xs), label='barriers')
        plt.title("barrier and minumum barriers distribution")
        plt.legend()
        plt.xlabel("B")
        plt.ylabel("P(B)")

        def animate(i):
            global ns

            if len(self.avalanche_sizes[i][1]) > len(self.avalanche_sizes[i][0]):
                xs = self.avalanche_sizes[i][1][:-1]
                xs_original = self.avalanche_sizes[i][1][:-1]
            ys = self.avalanche_sizes[i][0] / sum(self.avalanche_sizes[i][0])

            bin_size = xs[1] / 2.0

            pairs = [pair for pair in zip(xs, ys) if pair[1] != 0]
            xs = [pair[0] + bin_size for pair in pairs]
            ys = [pair[1] for pair in pairs]

            line.set_data(xs, ys)

            if len(xs) > 4:
                popt, pcov = curve_fit(powerlaw, xs, ys)

                line_fit.set_data(xs_original, powerlaw(xs_original, *popt))
                line_fit.set_label(r'$K=' + str(np.round(popt[1], 3)) + "$")
                coolax.legend()

            line_min.set_data(xs_space, self.barrier_dists[i][1](xs))
            line_bd.set_data(xs_space, self.barrier_dists[i][0](xs))

            # show variable 1

            # show variable 2

            # show the slums
            if len(self.states[i]) > ns:
                for slum, ax in zip(self.states[i][ns:len(self.states[i])],
                                    slumaxarr.flatten()[ns:len(self.states[i])]):
                    ims.append(ax.imshow(slum.ages, aspect='auto', cmap=cmap,
                                         interpolation='nearest', vmin=0, vmax=max_age))
                ns = len(self.states[i])

            plt.suptitle('iteration: ' + str(i * self.save_steps))
            for slum, im, in zip(self.states[i], ims):
                im.set_array(slum.ages)
            f.canvas.draw()
            if i == len(self.states) - 1:
                for im in ims:
                    im.set_array(np.ones((self.slum_size, self.slum_size)) * -1)

        _ = animation.FuncAnimation(f, animate, range(0, len(self.states)), interval=2, blit=False)
        plt.show()

def powerlaw(x, a, k):
    return np.power(a * np.array(x), -k)

def main():
    '''
    Runs a sample slum and shows different related plots.
    '''

    slums = Slums(4, (30, 30), empty_percent=0.06, time_limit=5000)

    slums.execute(save_steps=50)
    plt.close()
    slums.make_dashboard()
    # slums.plot_barrier_distribution()
    # slums.plot_avalanche_distance()
    # slums.plot_avalanche_size()
    # slums.plot_growth_over_time()
    # slums.plot_slums(start=0)


if __name__ == '__main__':
    main()
