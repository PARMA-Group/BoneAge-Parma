import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
matplotlib.use('Agg')

class Plotlight():
    def __init__(self):
        self.results = []
        self.labels = []
        self.colors = {"blue":"b", "red":"r","magenta":"m","green":"g"}

    
    def plot(self, name):
        temp = np.array(self.results)
        x = range(1, len(self.results)+1)
        colors = list(self.colors.keys())
        patches = []
        plt.figure()
        for i in range(len(self.results[0])):
            plt.plot(x, temp[:,i], color=self.colors[colors[i]])
            patches.append(mpatches.Patch(color=colors[i], label=self.labels[i]))

        plt.legend(handles=patches)
        plt.savefig(name)