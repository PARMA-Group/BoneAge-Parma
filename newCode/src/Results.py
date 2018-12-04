import csv

class Results:
    def __init__(self, x, y):
        self.loss = "loss"
        self.accuracy = "accuracy"
        self.train_results = []
        self.test_results = []

    def add_train_result(self, loss, acc):
        self.train_results.append({self.loss:loss, self.accuracy:acc})

    def add_test_result(self, loss, acc):
        self.test_results.append({self.loss:loss, self.accuracy:acc})

    # name can be train or test
    def make_csv(self, name):
        new_csv = open(name, "w")
        writter = csv.DictWriter(new_csv, fieldnames=[self.loss, self.accuracy])
        writter.writeheader()

        results = []
        if name == "train":
            results = self.train_results
        else:
            results = self.test_results
        for result in results:
            writter.writerow(result)
        new_csv.close()
