import numpy as np
import matplotlib.pyplot as plt

# Schwefel Function
def schwefel(x1, x2):
    return 418.9829*2 - (x1 * np.sin(np.sqrt(abs(x1)))) - (x2 * np.sin(np.sqrt(abs(x2))))

# Plot the function
def plot(max, min):
    x1 = x2 = np.linspace(min, max, num = 200)
    x1, x2 = np.meshgrid(x1, x2)
    z = schwefel(x1, x2)

    fig = plt.figure(figsize = (7, 7))
    ax = fig.add_subplot(111, projection = '3d')
    ax.plot_surface(x1, x2, z, cmap = 'viridis', alpha = .8)
    plt.title("Schwefel Function")
    plt.show()

plot(-500, 500)