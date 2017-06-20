from bs2d import BaxSneppen2D
import numpy as np

class Slums(object):

    def __init__(self, n_slums, slum_size=(15, 15)):
        slum_list = [BaxSneppen2D(np.random.rand(slum_size[0], slum_size[1])) for _ in range(n_slums)]

    def execute():
        pass

def main():
    pass

if __name__ == '__main__':
    main()