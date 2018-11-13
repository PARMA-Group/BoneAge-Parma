import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class Plotlight():
    def __init__(self):
        self.features = []
        self.labels = []
        self.colors = {"blue":"b", "red":"r","magenta":"m","green":"g"}

    
    def plot(self, name):
        temp = np.array(self.features)
        x = range(1, len(self.features)+1)
        colors = list(self.colors.keys())
        patches = []
        plt.figure()
        for i in range(len(self.features[0])):
            plt.plot(x, temp[:,i], color=self.colors[colors[i]])
            patches.append(mpatches.Patch(color=colors[i], label=self.labels[i]))

        plt.legend(handles=patches)
        plt.savefig(name)

#print(__file__)  

B = [[1,2,3],[1.5, 3, 6], [2, 4, 12]]
p = Plotlight()
p.features = B
p.labels = ["aaaa","bbbb","cccc"]
p.plot("asd.png")

#print(np.array([[1,1],[1,2],[1,3]])[:,0])

