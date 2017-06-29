'''
A slum network implementation based on 2D Bax-Sneppen models joined together in a network.

This code was built by Nicky Kessels, Maarten van der Sande, Wessel Klijnsma and Willem Vermeulen,
all master students Computational Science at the University of Amsterdam.
'''

from math import ceil
from copy import deepcopy
from bs2d import BaxSneppen2D
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch, Circle
import networkx as nx


class Slums(object):
    '''
    Encapsules a slum simulation based on 2D Bax-Sneppen models joined together in a network.
    '''

    # pylint: disable=too-many-instance-attributes, too-many-arguments

    # It's reasonable to have 12 variables, to be able to keep track of all parameters
    # and all slums.

    def __init__(self, n_slums, slum_size=(15, 15), empty_percent=0.25, random_select=False,
                 time_limit=10000, static_slums=False, static_people=False, strict_select=False,
                 cell_decrease_factor=0.8):
        # Set some overall parameters.
        self.time = 0
        self.time_limit = time_limit

        self.save_steps = 1

        self.slum_size = slum_size[0]
        self.static_slums = static_slums
        self.static_people = static_people

        self.random_select = random_select
        self.strict_select = strict_select
        assert not (random_select and strict_select),\
            "strict select can not be true when randomly selecting"

        self.threshold = 0
        self.n_slums = n_slums
        self.avalanche_size = [0]
        self.avalanche_sizes = []
        self.aval_start_val = 0

        # Create migration dict
        self.migrations = {}
        self.migrations['new'] = {}
        for i in range(n_slums):
            self.migrations[i] = {}
            self.migrations['new'][i] = 0
            for j in range(n_slums):
                self.migrations[i][j] = 0
        # The variable where the execute function will place the migration matrix in
        self.migration_matrix = None

        # Set some variables to keep track of all slums.
        self.slum_list = [BaxSneppen2D(slum_size, empty_percent,
                                       cell_decrease_factor) for _ in range(n_slums)]
        self.states = []

        self.previous_location = [(0, 0) for _ in range(n_slums)]
        self.distances = [[] for _ in range(n_slums)]
        self.barrier_dists = []

        self.new_person = self.get_new_person_time(0.2)
        self.migration_matrices = []

    def execute(self, moore=False, save_steps=25, net_freq=25):
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
        self.net_freq = net_freq

        # Keep updating the states until the time limit is reached.
        while self.update_state(moore):
            if iterator % save_steps == 0:
                self.states.append(deepcopy(self.slum_list))
                self.avalanche_sizes.append(deepcopy(self.avalanche_size))
                self.barrier_dists.append(self.get_barrier_distribution())

            iterator += 1

            if self.time % net_freq == 0:
                self.migration_matrix = np.zeros((len(self.migrations), len(self.migrations)))
                for key in self.migrations:
                    if key != 'new':
                        for inner_key in self.migrations[0]:
                            self.migration_matrix[key][inner_key] = self.migrations[key][inner_key]
                        self.migration_matrix[-1][key] = self.migrations['new'][key]

                self.migration_matrices.append(deepcopy(self.migration_matrix))
                self.reset_migration()

        self.migration_matrix = np.zeros((len(self.migrations), len(self.migrations)))
        for key in self.migrations:
            if key != 'new':
                for inner_key in self.migrations[0]:
                    self.migration_matrix[key][inner_key] = self.migrations[key][inner_key]
                self.migration_matrix[-1][key] = self.migrations['new'][key]

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

        # Add the migration to the migration array
        self.migrations[min_slum][to_slum] += 1

        slum_densities = [slum.get_density() for slum in self.slum_list]

        if not self.static_slums:
            if max(slum_densities) > 0.98 and min(slum_densities) > 0.5:
                print("New slum built.")
                self.slum_list.append(BaxSneppen2D((self.slum_size, self.slum_size),
                                                   empty_percent=1))
                self.previous_location.append((0, 0))
                self.distances.append([])

                self.migrations[len(self.slum_list) - 1] = {}
                self.migrations['new'][len(self.slum_list) - 1] = 0
                for i in range(len(self.slum_list)):
                    self.migrations[len(self.slum_list) - 1][i] = 0
                    self.migrations[i][len(self.slum_list) - 1] = 0
                self.n_slums += 1

        # migrate the person
        self.slum_list[to_slum].add_to_grid(min(min_vals))

        if not self.static_people:
            if self.time == self.new_person:
                # Add new people to the grid.
                print('New person added.')
                for _ in range(5):
                    to_slum = self.get_to_slum(min_slum)
                    self.migrations['new'][to_slum] += 1
                    self.slum_list[to_slum].add_to_grid()
                self.new_person = self.get_new_person_time(self.get_lambda())

        # Check if the time limit is reached, otherwise return False (and
        # the execution ends)
        if self.time > self.time_limit:
            return False

        self.time += 1
        return True

    def reset_migration(self):
        # Create migration dict
        self.migrations = {}
        self.migrations['new'] = {}
        for i in range(self.n_slums):
            self.migrations[i] = {}
            self.migrations['new'][i] = 0
            for j in range(self.n_slums):
                self.migrations[i][j] = 0
        # The variable where the execute function will place the migration matrix in
        self.migration_matrix = None

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

    def get_lambda(self):
        '''
        The frequency used in determining the waiting time.

        RETURNS
        ===================================================
        float
        A value for lambda used in our simulation.
        '''
        return (1 / 250) * max([slum.get_density() for slum in self.slum_list]) ** 2

    def get_new_person_time(self, lamb):
        '''
        Find the next time new persons are introduced to
        the simulation.

        PARAMETERS
        ===================================================
        lamb: float
        The lambda parameter used in the random
        exponential function.

        RETURNS
        ===================================================
        integer
        A new timestamp based on a random exponential
        distribution.
        '''
        wait = np.random.exponential(1 / lamb)
        return int(self.time + wait)

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

        pvalues = np.array(pvalues) ** 10

        # Normalise the pvalues and make a choice of a location for a cell to go to
        pvalues = pvalues / np.sum(pvalues)

        if self.strict_select:
            return np.argmax(pvalues)
        else:
            return np.random.choice(range(len(self.slum_list)), 1, p=pvalues)[0]

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
        total = sum(pvalues)
        pvalues = [pvalue / total for pvalue in pvalues]

        return np.random.choice(range(len(self.slum_list)), 1, p=pvalues)[0]

    def plot_slums(self, start=0, reset=True):
        '''
        Plots all simulated slums.

        PARAMETERS
        ===================================================
        start: integer
        The number of the timestep at which the animation
        has to be started.
        '''

        cols = ceil(len(self.slum_list) ** 0.5)
        rows = ceil(len(self.slum_list) / cols)

        figure, axarr = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True)

        # Remove the axes labels and set spacing.
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.xticks([])
        plt.yticks([])

        # Set the colour map.
        cmap = get_colormap()

        # Calculate the maximum age for the plot.
        max_age = max([np.max(slum.ages) for slum in self.slum_list])

        # Initialise the plot.
        imgs = list()

        if len(self.slum_list) == 1:
            axarr = np.array([axarr])

        # Display a map of the ages within each slum.
        n_slums = len(self.states[0])
        for slum, axes in zip(self.states[0], axarr.flatten()):
            imgs.append(axes.imshow(slum.ages, aspect='auto', cmap=cmap, interpolation='nearest',
                                    vmin=0, vmax=max_age))

        def animate(i):
            '''
            Used to animate the maps of cells ages of each
            slum.

            PARAMETERS
            ===================================================
            i: integer
            The frame to display.

            RETURNS
            ===================================================
            imgs: list of plt.subplots
            A list of all slum plots in use.
            '''

            nonlocal n_slums, reset

            # If there is a new slum, add it to the output.
            if len(self.states[i]) > n_slums:
                for slum, axes in zip(self.states[i][n_slums:len(self.states[i])],
                                      axarr.flatten()[n_slums:len(self.states[i])]):
                    imgs.append(axes.imshow(slum.ages, aspect='auto', cmap=cmap,
                                            interpolation='nearest', vmin=0, vmax=max_age))
                n_slums = len(self.states[i])

            plt.suptitle('iteration: ' + str(i * self.save_steps))

            # Add information on the current state to the images.
            for slum, img, in zip(self.states[i], imgs):
                img.set_array(slum.ages)

            figure.canvas.draw()

            # Reset the plot when done!
            if i == len(self.states) - 1 and reset:
                for img in imgs:
                    img.set_array(np.ones((self.slum_size, self.slum_size)) * -1)

            return imgs

        # Start the animation.
        _ = animation.FuncAnimation(figure, animate, range(int(len(self.states) * start),
                                                           len(self.states), 1), interval=2,
                                    blit=False)
        plt.show()

    def plot_barrier_distribution(self):
        '''
        Plot the barrier distributions.
        '''

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
        plt.xlabel(r"$B$")
        plt.ylabel(r"$P(B)$")
        plt.show()

    def get_barrier_distribution(self):
        '''
        Retrieve the barrier distributions.

        RETURNS
        ===================================================
        (min_density, bar_density) =
        (scipy.stats.kde.gaussian_kde,
         scipy.stats.kde.gaussian_kde)
        The estimated gaussian densities of the minima
        and barriers found.
        '''

        barriers = []
        minima = []

        for slum in self.slum_list:
            barriers = barriers + list(slum.state[np.where(slum.state <= 1)].flatten())

        for timestep in self.states:
            minima.append(min([state.get_min_val() for state in timestep]))

        if len(minima) == 1:
            minima.append(0)

        minima = np.array(minima)
        barriers = np.array(barriers)
        min_density = gaussian_kde(minima)
        bar_density = gaussian_kde(barriers)

        return (min_density, bar_density)

    def plot_avalanche_distance(self):
        '''
        Plot the distances between succesive mutations.
        '''

        avalanches = []
        for each in self.distances:
            avalanches = avalanches + each

        (counts, bins, _) = plt.hist(avalanches, bins=100)
        plt.clf()
        plt.loglog(bins[:-1], counts / sum(counts))

        plt.title("distance between succesive mutations")
        plt.xlabel(r"$log_{10}(X)$")
        plt.ylabel(r"$log_{10}(C(X))$")
        plt.show()

    def plot_avalanche_size(self):
        '''
        Plot the avalanche sizes.
        '''

        # (counts, bins, _) = plt.hist(self.avalanche_size, bins=100)
        # plt.clf()
        # plt.loglog(bins[:-1], counts / sum(counts))

        # plt.title("avalanche sizes")
        # plt.xlabel(r"S$")
        # plt.ylabel(r"P(S)$")
        # plt.show()


    def plot_growth_over_time(self):
        '''
        Plot the growth of slums over time
        '''

        growths = [[] for _ in range(len(self.states[-1]))]
        scaler = self.save_steps

        for state in self.states:
            # pylint: disable=consider-using-enumerate
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

    def setup_slum_anim(self, cmap, max_age):
        '''
        Initialises all grids used in the slum animation.

        PARAMETERS
        ===================================================
        cmap: matplotlib.colors.LinearSegmentedColormap
        The colour map used.

        max_age: integer
        The maximum age of any cell within the slums.

        RETURNS
        ===================================================
        figure: matplotlib.figure.Figure
        The figure used in the animation.

        imgs: list
        A list of all slum age images.

        rows: integer
        The number of rows.

        cols: integer
        The number of columns.

        n_slums: integer
        The number of slums.

        slumaxarr: list
        A lis of all axes of all slums.
        '''

        slumaxarr = []
        size = len(self.states[-1])
        n_slums = len(self.states[0])
        cols = ceil(size ** 0.5) #+ 1
        rows = ceil(size / cols)

        figure = plt.figure(figsize=(14, 9))

        for i in range(size):
            slumaxarr.append(plt.subplot2grid((0 + rows, cols), (i // cols , (i % cols))))

        for i, slumaxes in enumerate(slumaxarr):
            slumaxes.set_xticklabels([])
            slumaxes.set_yticklabels([])

        imgs = list()

        for slum, axes in zip(self.states[0], slumaxarr):
            imgs.append(axes.imshow(slum.state, aspect='auto', cmap=cmap, interpolation='nearest',
                                    vmin=0, vmax=max_age))

        return figure, imgs, rows, cols, n_slums, slumaxarr

    # pylint: disable=too-many-locals
    def make_dashboard(self, show_powerlaw=False):
        '''
        Show a dashboard of all plots and two figures containing information on the avalanche
        sizes and barrier distributions.
        '''

        cmap = get_colormap()
        max_age = max([np.max(slum.ages) for slum in self.slum_list])
        max_age = 1
        # n_slums is used in the nested animate function.
        # pylint: disable=unused-variable
        figure, imgs, rows, cols, n_slums, slumaxarr = self.setup_slum_anim(cmap, max_age)
        x_list = [x for x in sorted(list(set(self.avalanche_sizes[-1]))) if x != 0]
        y_list = [self.avalanche_sizes[-1].count(x) for x in x_list]

        bound = -1

        for i in range(len(x_list) - 1):
            if x_list[i + 1] - x_list[i] > 5:
                bound = i

        if bound > 0:
            x_list = x_list[:bound]
            y_list = y_list[:bound]

        # Plot the avalanche sizes.
        # pwax = plt.subplot2grid((1 + rows, cols), (rows, 0))
        # line, = pwax.loglog(x_list, y_list, ".")

        # Plot the powerlaw based on the avalanche sizes.
        # popt, _ = curve_fit(powerlaw, x_list, y_list, bounds=((0, 0), (np.inf, 6)))
        #
        # power_list = [y for y in powerlaw(x_list, *popt) if y > 1]
        #
        # line_fit, = pwax.plot(x_list[:len(power_list)], power_list, 'r-',
        #                       label=r'$K=' + str(np.round(popt[1], 3)) + "$")
        #
        # plt.title("avalanche sizes")
        # plt.legend()
        # plt.xlabel(r"$S$")
        # plt.ylabel(r"$P(S)$")

        # Plot the barrier distributions
        # x_space = np.linspace(0, 1, 300)
        # bdax = plt.subplot2grid((1 + rows, cols), (rows, 1))
        # bdax.set_yticklabels([])
        # bdax.set_xlabel(r"$B$")
        # bd2ax = bdax.twinx()
        # bd2ax.set_yticklabels([])
        # line_min, = bdax.plot(x_space, self.barrier_dists[-1][0](x_space), label='minima')
        # bdax.set_ylabel(r"$P(B)$")
        # line_bd, = bd2ax.plot(x_space, self.barrier_dists[-1][1](x_space), color='g', label='barriers')
        # plt.title("barrier and minumum barriers dist (R, P)")
        #plt.legend()

        # bd2ax.set_ylabel(r"$R(B)$")


        # plot the density over time
        densities = [[] for _ in range(len(self.states[-1]))]
        for i in range(len(self.states[-1])):
            for j in range(len(self.states)):
                if len(self.states[j]) >= (i + 1):
                    densities[i].append(self.states[j][i].get_density())
                else:
                    densities[i].append(0)

        # denax = plt.subplot2grid((1 + rows, cols), (rows, 2))
        # denax.set_xlim([0, 1])
        # denax.set_ylim([0, 1])
        # plt.title("Slum densities")
        # plt.ylabel("density")
        # plt.xlabel("time")
        lines = []

        # for _ in range(len(self.states[-1])):
        #     lin, = denax.plot(np.linspace(0, 1, len(self.states)))
        #     lines.append(lin)

        def animate(i):
            '''
            Used to animate the maps of cells ages of each
            slum, as well as both graphs.

            PARAMETERS
            ===================================================
            i: integer
            The frame to display.
            '''

            nonlocal n_slums, cmap, max_age, show_powerlaw#, pwax

            # Set the x coordinate to the middle of the bin.
            # x_list = [x for x in sorted(list(set(self.avalanche_sizes[i]))) if x != 0]
            # y_list = [self.avalanche_sizes[i].count(x) for x in x_list]
            #
            # line.set_data(x_list, y_list)

            # Try to fit a power law.
            # if len(x_list) > 4 and show_powerlaw:
            #     bound = -1
            #
            #     for i in range(len(x_list) - 1):
            #         if x_list[i + 1] - x_list[i] > 5:
            #             bound = i
            #
            #     try:
            #         if bound > 0:
            #             popt, _ = curve_fit(powerlaw, x_list[:bound], y_list[:bound], bounds=((0, 0), (np.inf, 6)))
            #         else:
            #             popt, _ = curve_fit(powerlaw, x_list, y_list, bounds=((0, 0), (np.inf, 6)))
            #
            #         line_fit.set_data(x_list[:bound], powerlaw(x_list[:bound], *popt))
            #         line_fit.set_label(r'$K=' + str(np.round(popt[1], 3)) + "$")
            #         pwax.legend()
            #     except RuntimeError:
            #         pass

            # Plot the barrier distributions
            # x_space = np.linspace(0, 1, 300)
            #
            # line_min.set_data(x_space, self.barrier_dists[i][0](x_space))
            # line_bd.set_data(x_space, self.barrier_dists[i][1](x_space))
            #
            # # plot the density of the slums over time
            # for j, lin in enumerate(lines):
            #     lin.set_data(np.linspace(0, 1, len(densities[j]))[:i], densities[j][:i])
            #
            # # Show the slums.
            # if len(self.states[i]) > n_slums:
            #     for slum, axes in zip(self.states[i][n_slums:len(self.states[i])],
            #                           slumaxarr[n_slums:len(self.states[i])]):
            #         imgs.append(axes.imshow(slum.ages, aspect='auto', cmap=cmap,
            #                                 interpolation='nearest', vmin=0, vmax=max_age))
            #
            #     n_slums = len(self.states[i])

            plt.suptitle('iteration: ' + str(i * self.save_steps))

            for slum, img, in zip(self.states[i], imgs):
                img.set_array(slum.state)

            figure.canvas.draw()

            if i == len(self.states) - 1:
                for img in imgs:
                    img.set_array(-np.ones((self.slum_size, self.slum_size)))

        figure.subplots_adjust(wspace=0.44)
        plt.savefig('../docs/videos/barebones_fitness.png')
        ani = animation.FuncAnimation(figure, animate, range(int(0.9*len(self.states)), len(self.states)), interval=2,
                                    blit=False)
        ani.save('../docs/videos/barebones_fitness.gif', writer='imagemagick')
        plt.show()

    def plot_network(self):
        figure = plt.figure(figsize=(14, 9))

        def animate(i):
            G = nx.from_numpy_matrix(self.migration_matrices[i][:-1, :-1], create_using=nx.DiGraph())
            layout = nx.circular_layout(G)
            figure.clf()
            ax = figure.gca()

            ax.set_ylim(-1.25, 1.25)
            ax.set_xlim(-1.25, 1.25)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            plt.title('iteration: ' + str(i * self.net_freq))
            edge_labels = {}

            # Determine the maximum edge weight and labels
            max_weight = 0
            for (u, v, d) in G.edges(data=True):
                if d['weight'] > max_weight and u != v:
                    max_weight = d['weight']

            bbox_opts = dict(alpha=0, lw=0)

            # Draw all the nodes
            nx.draw_networkx_labels(G, pos=layout, font_size=30, font_color='r', font_weight='bold')

            # Draw some curved edges
            seen = {}

            for (u, v, d) in G.edges(data=True):
                if u == v:
                    ax.text(layout[u][0] - (len(str(d['weight'])) - 1) * 0.006, layout[u][1] - 0.12,
                            "Self: " + str(int(d['weight'])), fontsize=15)
                    continue

                n1 = layout[u]
                n2 = layout[v]
                rad = 0.1

                if (u, v) in seen:
                    rad = seen.get((u, v))
                    rad = (rad + np.sign(rad) * 0.1) * -1

                e = FancyArrowPatch(n1, n2, arrowstyle='fancy',
                                    connectionstyle='arc3,rad=%s' % rad,
                                    mutation_scale=d['weight'] / max_weight * 25,
                                    lw=2,
                                    alpha=0.2,
                                    color='g')
                seen[(u, v)] = rad
                ax.add_patch(e)

            for (u, v, d) in G.edges(data=True):
                if u == v:
                    G.remove_edge(u, v)
                else:
                    edge_labels[(u, v)] = int(d['weight'])

            nx.draw_networkx_edge_labels(G, pos=layout, edge_labels=edge_labels, label_pos=0.9, font_size=16,
                                         bbox=bbox_opts)
            if i == 0:
                plt.savefig('../docs/videos/slum_network_np.png')


        ani = animation.FuncAnimation(figure, animate, range(0, len(self.migration_matrices)), interval=200,
                                    blit=False)
        ani.save('../docs/videos/slum_network_np.gif', writer='imagemagick')
        plt.show()

# x, a and k are commonly used variables in a powerlaw distribution.
# pylint: disable=invalid-name
def powerlaw(x, a, k):
    '''
    Returns a powerlaw of input x for parameters a and
    k.

    PARAMETERS
    ===================================================
    x: numpy.ndarray
    An array of input x.

    a: integer
    Constant in multiplication.

    k: integer
    Power within the powerlaw.

    RETURNS
    ===================================================
    numpy.ndarray
    An array of powerlaw values of the input x for the
    corresponding parameters.
    '''

    return np.power(a * np.array(x), -k)


def get_colormap():
    '''
    Returns the colour map used in the slum network
    simulation display.

    PARAMETERS
    ===================================================
    None

    RETURNS
    ===================================================
    matplotlib.colors.LinearSegmentedColormap
    The colour map used.
    '''

    # pylint: disable=maybe-no-member
    cmap = plt.cm.jet
    cmap.set_over((1, 1, 1, 1))

    return cmap


def main():
    '''
    Runs a sample slum and shows different related plots.
    '''
    # plt.xkcd()
    slums = Slums(1, (60, 60), empty_percent=0.1, time_limit=40000, static_people=True, static_slums=True)
    # nrofslums_parameter_plot(np.linspace(1,5,5), 10, 1000)
    # singleslumsize_parameter_plot(np.linspace(5,50,10), 10, 1000)

    #plt.rcParams.update({'font.size': 14})
    # empty_percent_parameter_plot(10, 10, 20000)
    # singleslumsize_parameter_plot(np.linspace(5,50,10), 20, 25000)
    # nrofslums_parameter_plot(np.linspace(1,8,8), 10, 20000)
    # effect_of_location(10, 20000)
    # cell_decrease_factor_plot(np.linspace(0.1, 1, 10), 30, 20000)

    slums.execute(save_steps=20, net_freq=50)
    #slums.plot_network()
    # slums.plot_network()

    print('Simulation has ended.')
    # slums.plot_network()
    slums.make_dashboard()

    # slums.plot_barrier_distribution()
    # slums.plot_avalanche_distance()
    # slums.plot_avalanche_size()
    # slums.plot_growth_over_time()
    # slums.plot_slums(start=0)


if __name__ == '__main__':
    main()
