import csv

class Results:
    def __init__(self, path):
        self.path = path
        self.loss = "loss"
        self.accuracy = "mae"
        self.train_results = []
        self.test_results = []

    def add_train_result(self, loss):
        self.train_results.append({self.loss:loss})

    def add_test_result(self, loss, acc):
        self.test_results.append({self.loss:loss, self.accuracy:acc})

    def write_results(self):
        self.make_csv("train")
        self.make_csv("test")

    # name can be train or test
    def make_csv(self, name):
        new_csv = open(self.path + name, "w")

        fields = [self.loss, self.accuracy]
        results = []
        if name == "train":
            results = self.train_results
            fields = [self.loss]
        else:
            results = self.test_results
            
        writter = csv.DictWriter(new_csv, fieldnames=fields)
        writter.writeheader()
        
        for result in results:
            writter.writerow(result)
        new_csv.close()
