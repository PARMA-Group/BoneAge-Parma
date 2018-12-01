import csv

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
