from bs2d import BaxSneppen2D
import numpy as np

class Slums(object):

    def __init__(self, n_slums, slum_size=(15, 15)):
        self.slum_list = [BaxSneppen2D(slum_size) for _ in range(n_slums)]

    def execute(self, moore=False):
        while self.update_state(moore):
            continue

    def update_state(self, moore=False):
        min_vals = [slum.get_min_val() for slum in self.slum_list]

        self.slum_list[np.argmin(min_vals)].update_state(moore)

        # TODO Modify this part to make some network structure

        self.slum_list[(np.argmin(min_vals) + 1) % len(self.slum_list)].add_to_grid(min(min_vals))

def main():
    slums = Slums(4)
    slums.execute()

if __name__ == '__main__':
    main()