from bs2d import BaxSneppen2D
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import matplotlib.animation as animation

class Slums(object):

    def __init__(self, n_slums, slum_size=(15, 15)):
        self.slum_list = [BaxSneppen2D(slum_size) for _ in range(n_slums)]
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

        # TODO Modify this part to make some network structure

        if not self.slum_list[(np.argmin(min_vals) + 1) % len(self.slum_list)].add_to_grid(min(min_vals)):
            self.slum_list[(np.argmin(min_vals) + 2) % len(self.slum_list)].add_to_grid(min(min_vals))

        if self.time > 1000:
            return False

        self.time += 1
        return True

    def plot_slums(self):
        f, axarr = plt.subplots(4, sharex=True, sharey=True)
        ims = list()
        max_ages = [np.max(slum.ages) for slum in self.slum_list]
        max_age = max(max_ages)
        for slum, ax in zip(self.slum_list, axarr):
            ims.append(ax.imshow(slum.ages, aspect='auto', cmap='jet_r', interpolation='nearest', vmin=0, vmax=max_age))

        def animate(i):
            plt.title('iteration: ' + str(i))
            for slum, im, in zip(self.states[i], ims):
                im.set_array(slum.ages)
            f.canvas.draw()
            return ims

        ani = animation.FuncAnimation(f, animate, range(int(len(self.states) * 0), len(self.states)), interval=2, blit=False)
        plt.show()




def main():
    slums = Slums(4, (50,50))
    slums.execute()

if __name__ == '__main__':
    main()
