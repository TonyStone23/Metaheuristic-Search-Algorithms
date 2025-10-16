import numpy as np
from math import pi
import matplotlib.pyplot as plt

def rastrigin(x1, x2):
    return 20 + sum([x1**2 - 10 * np.cos(2 * pi * x1) + x2**2 - 10 * np.cos(2 * x2)])

def plot(min, max):

    x1 = x2 = np.linspace(min, max)
    x1, x2 = np.meshgrid(x1, x2)
    z = rastrigin(x1, x2)

    fig = plt.figure(figsize = (7, 7))
    ax = fig.add_subplot(111, projection = '3d')
    ax.plot_surface(x1, x2, z, cmap = 'viridis', alpha = .8)
    plt.title("Rastrigin Function")
    plt.show()

plot(-5, 5)