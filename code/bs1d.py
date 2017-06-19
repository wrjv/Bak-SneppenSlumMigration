#
# Bax-Sneppen 1D implementation by
# Nicky, Maarten, Wessel and Willem
#

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


class BaxSneppen1D(object):

    def __init__(self, initial_values):
        self.states = [initial_values]
        self.ages = [np.zeros(len(initial_values))]
        self.time = 0

        self.prev_loc = 0
        self.distances = []

        self.avelanches = []
        self.cur_av_count = 0
        self.cur_av_start = -1

        self.update_freq = 100

    def execute(self):
        while self.update_state():
            continue

        # Write down the last avelanche
        self.avelanches.append(self.cur_av_count)

        print(self.ages[-1])
        print(len(self.states))

    def update_state(self):
        new_ages = deepcopy(self.ages[-1])

        # Build a new state
        new_state = deepcopy(self.states[-1])

        for i in range(self.update_freq):
            new_ages = new_ages + 1
            min_val = np.argmin(new_state)

            # Count avelanches
            if new_state[min_val] > self.cur_av_start:
                if self.cur_av_start != -1:
                    self.avelanches.append(self.cur_av_count)
                self.cur_av_count = 0
                self.cur_av_start = new_state[min_val]
            else:
                self.cur_av_count += 1

            # calculate the distance between succesive mutations
            self.distances.append(np.min(np.abs([self.prev_loc - min_val, min_val - self.prev_loc])))
            self.prev_loc = min_val

            # Stopping criterium
            #if new_state[min_val] > 0.67:
            #    return False
            if len(self.states) > 10000:
                return False

            # Modify the values around the minimum value
            new_state[min_val - 1] = np.random.uniform(0, 1, 1)
            new_state[min_val] = np.random.uniform(0, 1, 1)
            new_state[(min_val + 1) % len(new_state)] = np.random.uniform(0, 1, 1)

            # Modify the cell ages
            new_ages[min_val - 1] = 0
            new_ages[min_val] = 0
            new_ages[(min_val + 1) % len(new_state)] = 0

        self.states.append(new_state)
        self.ages.append(new_ages)
        self.time += 1
        return True

    def plot_ages(self):
        plt.imshow(self.ages[::100], aspect='auto', cmap='jet_r', interpolation='nearest')
        plt.show()

    def barrier_distribution(self):
        plt.hist(self.states[-1], bins=30)
        plt.show()

    def distance_distribution(self):
        self.distances.pop(0)
        counter = {}
        # print(len(self.states[0]))
        for i in range(0,len(self.states[0])):
            counter[i] = 0
        for val in self.distances[int(0.35*len(self.distances)):]:
            counter[val] += 1
        keys = list(counter.keys())
        print(np.sum([counter[key] for key in keys]))
        plt.loglog(keys, [counter[key] for key in keys])
        plt.show()

    def plot_avelanches(self):
        plt.plot(self.avelanches)
        plt.show()

def main():
    initial_values = np.random.rand(4096)
    bs1d = BaxSneppen1D(initial_values)
    bs1d.execute()
    bs1d.plot_avelanches()
    #bs1d.plot_ages()
    #bs1d.barrier_distribution()
    #bs1d.distance_distribution()
    #print(len(bs1d.distances))

if __name__ == '__main__':
    main()
