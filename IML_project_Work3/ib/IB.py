import numpy as np
import time


class IB:
    def __init__(self, algorithm):
        self.algorithm_mapping = {'ib1': self.ib1, 'ib2': self.ib2, 'ib3': self.ib3}  # algorithm mappings

        self.algorithm = self.algorithm_mapping[algorithm]

        self.cd = []  # concept description
        self.cd_classes = []  # concept description labels
        self.correct_num = 0  # correctly classified data
        self.length = 0  # dataset length
        self.time = 0  # execution time

    def fit_and_predict(self, dataset, class_labels):
        self.correct_num = 0
        self.length = len(dataset)
        self.cd.append(dataset[0])  # append first data in cd because at the start it is empty
        self.cd_classes.append(class_labels[0])  # append first label in the cd because at the start it is empty
        start_time = time.perf_counter()  # define starting time

        self.algorithm(dataset, class_labels)  # fit and predict data

        end_time = time.perf_counter()  # define ending time
        self.time = end_time - start_time  # calculate the overall time

    def ib1(self, dataset, class_labels):

        # zip data and labels and iterate over them
        for index, (data_point, label) in enumerate(zip(dataset, class_labels)):

            # find the index of most similar one
            min_index = self.find_similar(data_point)

            # if it is correctly classified increase the number
            if label == self.cd_classes[min_index]:
                self.correct_num += 1

            # because we already have first data point in cd we are not adding it any more
            if index != 0:
                self.cd.append(data_point)
                self.cd_classes.append(label)

    def ib2(self, dataset, class_labels):
        # zip data and labels and iterate over them
        for index, (data_point, label) in enumerate(zip(dataset, class_labels)):

            # find the index of most similar one
            min_index = self.find_similar(data_point)

            # if it is correctly classified increase the number
            if label == self.cd_classes[min_index]:
                self.correct_num += 1
            # if it is incorrectly classified add in the cd
            else:
                self.cd.append(data_point)
                self.cd_classes.append(label)

    def ib3(self, dataset, class_labels):
        # zip data and labels and iterate over them
        for index, (data_point, label) in enumerate(zip(dataset, class_labels)):
            pass

    def find_similar(self, data_point):
        # find euclidean distance between data point and entire cd
        distance_matrix = np.linalg.norm(np.array(self.cd) - data_point, axis=1)

        # find the index of the most similar one
        min_index = np.argmin(distance_matrix)
        return min_index

    def calculate_accuracy(self):
        return self.correct_num / self.length

    def print_results(self):
        print(f"accuracy: {self.calculate_accuracy() * 100} %")
        print(f"execution time: {self.time}")
        print(f"used memory: {len(self.cd)} / {self.length}")



