import csv
import torch
import numpy as np
import _pickle as pickle
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class LogCSV:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.results = []

    def add_result(self, x, y):
        self.results.append({self.x:x, self.y:y})

    def make_csv(self, name):
        new_csv = open(name, "w")
        writter = csv.DictWriter(new_csv, fieldnames=[self.x, self.y])
        writter.writeheader()
        print("* Creando CSV de la forma {},{} *".format(self.x, self.y))
        for i in self.results:
            writter.writerow(i)            
        new_csv.close()

    def keras_plot_file(self, file1, file2, title):
        # este es training
        f1 = open(file1,"rb")
        emp1 = pickle.load(f1)


        f2 = open(file2,"rb")
        emp2 = pickle.load(f2)

        keys = list(emp1.keys())
        key1 = keys[0]
        key2 = keys[1]
        datax1 = emp1[key1]
        datax2 = emp2[key1]
        datay1 = emp1[key2]        
        datay2 = emp2[key2]
        x = range(1, len(datax1) + 1)

        plt.figure()
        plt.plot(x, datax1,"r")
        plt.plot(x, datax2,"b")
        plt.legend(["DNLM", "REGISTERED"], loc='upper right')
        plt.title(title + " " + key1)
        plt.xlabel("Epochs")
        plt.ylabel("%")
        plt.savefig(title + key1 + "_PLOT.png")

        plt.figure()
        plt.plot(x, datay1,"r")
        plt.plot(x, datay2,"b")
        plt.legend(["DNLM", "REGISTERED"], loc='upper right')
        plt.title(title + " " + key2)
        plt.xlabel("Epochs")
        plt.ylabel("%")
        plt.savefig(title + key2 + "_PLOT.png")        

        """
        plt.figure()
        plt.plot(x, data1,"r")
        plt.plot(x, data2,"b")
        plt.legend([key1, key2], loc='upper right')
        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel("%")
        plt.savefig(title + "_PLOT.png")
        """

    def dual_plot(self, file1, file2):

        data1 = genfromtxt(file1, delimiter=',', skip_header = 1)
        data2 = genfromtxt(file2, delimiter=',', skip_header = 1)

        data1_y1 = data1[:,0]
        data2_y1 = data2[:,0]

        data1_y2 = data1[:,1]
        data2_y2 = data2[:,1]

        x = range(1, data1_y1.shape[0] + 1)
        
        plt.figure()
        plt.plot(x, data1_y1,"r")
        plt.plot(x, data2_y1,"b")
        plt.legend(["DNLM", "Registered"], loc='upper right')
        plt.title(self.x)
        plt.xlabel("Epochs")
        plt.ylabel("Meses")
        plt.savefig(self.x + "_PLOT.png")

        plt.figure()
        plt.plot(x, data1_y2, "r")
        plt.plot(x, data2_y2, "b")
        plt.title(self.y)
        plt.xlabel("Epochs")
        plt.ylabel("Meses")
        plt.legend(["DNLM", "Registered"], loc='upper right')
        plt.savefig(self.y + "_PLOT.png")

csv1 = "malednlm_performance.csv"
csv2 = "male_performance.csv"
log = LogCSV("L1Loss","RMSE")
log.keras_plot_file("dnlm0m_training_results.p", "nodnlm0m_training_results.p", "Male")
log.keras_plot_file("dnlm0f_training_results.p", "nodnlm0f_training_results.p", "Female")
#log.keras_plot_file("training_results_f.p", "Retraining Training Results Female")