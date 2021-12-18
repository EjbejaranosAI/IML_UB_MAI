import math
import random

import numpy as np
import time
from scipy.spatial import distance


class IB:
    def __init__(self, algorithm):
        self.algorithm_mapping = {'ib1': self._ib1, 'ib2': self._ib2, 'ib3': self._ib3}  # algorithm mappings

        self.algorithm = self.algorithm_mapping[algorithm]

        self.cd = []  # concept description
        self.cd_classes = []  # concept description labels
        self.correct_num = 0  # correctly classified data
        self.dataset_size = 0  # dataset dataset_size
        self.total_size = 0
        self.time = 0  # execution time


        # those are for ib3
        self.accuracies = {}
        self.frequencies = {}

    def fit_and_predict(self, dataset, class_labels):
        self.correct_num = 0
        self.dataset_size = len(dataset)
        self.total_size += len(dataset)
        self.cd.append(dataset[0])  # append first data in cd because at the start it is empty
        self.cd_classes.append(class_labels[0])  # append first label in the cd because at the start it is empty

        # for ib3 only
        # init first instance of cds accuracy metric
        self.accuracies[str(dataset[0])] = {'correct': 0, 'incorrect': 0}
        # increase the number of frequency for first instance in CD
        self.frequencies[class_labels[0]] = 1


        start_time = time.perf_counter()  # define starting time

        self.algorithm(dataset, class_labels)  # fit and predict data

        end_time = time.perf_counter()  # define ending time
        self.time = end_time - start_time  # calculate the overall time



    def _ib1(self, dataset, class_labels):

        # zip data and labels and iterate over them
        for index, (data_point, label) in enumerate(zip(dataset[1:], class_labels[1:])):

            # find the index of most similar one
            min_index = self._find_similar(data_point, self.cd)

            # if it is correctly classified increase the number
            if label == self.cd_classes[min_index]:
                self.correct_num += 1

            # because we already have first data point in cd we are not adding it any more

            self.cd.append(data_point)
            self.cd_classes.append(label)

    def _kib1(self, dataset, class_labels,k):

        # zip data and labels and iterate over them
        for index, (data_point, label) in enumerate(zip(dataset[1:], class_labels[1:])):

            # find the index of most similar one
            min_indices = self._find_k_similar(data_point, self.cd,k)
            min_index = min_indices[:1]
            most_freq = self.most_voted(min_indices)


            # if it is correctly classified increase the number
            if label == most_freq:
                self.correct_num += 1

            # because we already have first data point in cd we are not adding it any more

            self.cd.append(data_point)
            self.cd_classes.append(label)

    def _ib2(self, dataset, class_labels):
        # zip data and labels and iterate over them
        for index, (data_point, label) in enumerate(zip(dataset[1:], class_labels[1:])):

            # find the index of most similar one
            min_index = self._find_similar(data_point, self.cd)

            # if it is correctly classified increase the number
            if label == self.cd_classes[min_index]:
                self.correct_num += 1
            # if it is incorrectly classified add in the cd
            else:
                self.cd.append(data_point)
                self.cd_classes.append(label)

    def _ib3(self, dataset, class_labels):

        number_of_instances_processed = self.total_size - self.dataset_size

        # zip data and labels and iterate over them and skip the first one because we already added them in CD
        for length, (data_point, label) in enumerate(zip(dataset[1:], class_labels[1:])):

            if label not in self.frequencies:
                self.frequencies[label] = 0

            # length + 1 because we skipped the first one
            acceptable_instances, acceptable_instances_labels = self._get_acceptable_instances(0.9, number_of_instances_processed + length + 1)

            if len(acceptable_instances) == 0:
                acceptable_instance_index = random.randint(0, len(self.cd) - 1)
                acceptable_instance = self.cd[acceptable_instance_index]
                acceptable_instance_label = self.cd_classes[acceptable_instance_index]
            else:
                similar_index = self._find_similar(data_point, acceptable_instances)
                acceptable_instance = acceptable_instances[similar_index]
                acceptable_instance_label = acceptable_instances_labels[similar_index]

            if label == acceptable_instance_label:
                self.accuracies[str(acceptable_instance)]['correct'] += 1
                self.frequencies[label] += 1
                self.correct_num += 1
            else:
                self.accuracies[str(acceptable_instance)]['incorrect'] += 1
                self.cd.append(data_point)
                self.cd_classes.append(label)

                self.accuracies[str(data_point)] = {'correct': 0, 'incorrect': 0}

            # length + 2 because we already classified
            self.cd, self.cd_classes = self._update_cd(self.frequencies, 0.7, number_of_instances_processed + length + 2)

    def _calculate_boundaries(self, p, z, n):
        min_boundary = (p + z ** 2 / (2 * n) - z * (math.sqrt((p * (1 - p) / n) + (z ** 2 / (4 * n ** 2))))) / (1 + z ** 2 / n)
        max_boundary = (p + z ** 2 / (2 * n) + z * (math.sqrt((p * (1 - p) / n) + (z ** 2 / (4 * n ** 2))))) / (1 + z ** 2 / n)
        return [min_boundary, max_boundary]

    def _get_acceptable_instances(self, z, n):
        acceptable_instances = []
        acceptable_instances_labels = []
        for cd_point, cd_label in zip(self.cd, self.cd_classes):
            try:
                accuracy_p = self.accuracies[str(cd_point)]['correct'] / (
                        self.accuracies[str(cd_point)]['correct'] + self.accuracies[str(cd_point)]['incorrect'])
            except ZeroDivisionError:
                accuracy_p = 0
            instance_accuracy = self._calculate_boundaries(accuracy_p, z, n)

            frequency_p = self.frequencies[cd_label] / n
            class_observed_frequency = self._calculate_boundaries(frequency_p, z, n)

            if instance_accuracy[0] > class_observed_frequency[1]:
                acceptable_instances.append(cd_point)
                acceptable_instances_labels.append(cd_label)

        return acceptable_instances, acceptable_instances_labels

    def _update_cd(self, frequencies, z, n):
        new_cd = []
        new_cd_labels = []
        for cd_point, cd_label in zip(self.cd, self.cd_classes):
            try:
                accuracy_p = self.accuracies[str(cd_point)]['correct'] / (
                        self.accuracies[str(cd_point)]['correct'] + self.accuracies[str(cd_point)]['incorrect'])
            except ZeroDivisionError:
                accuracy_p = 0
            instance_accuracy = self._calculate_boundaries(accuracy_p, z, n)

            frequency_p = frequencies[cd_label] / n
            class_observed_frequency = self._calculate_boundaries(frequency_p, z, n)

            # dont remove and save
            if not instance_accuracy[1] < class_observed_frequency[0]:
                new_cd.append(cd_point)
                new_cd_labels.append(cd_label)
            else:
                # remove from accuracies
                del self.accuracies[str(cd_point)]
        return new_cd, new_cd_labels

    def _find_similar(self, data_point, cd):
        # find euclidean distance between data point and entire cd
        distance_matrix = np.linalg.norm(np.array(cd) - data_point, axis=1)

        # find the index of the most similar one
        min_index = np.argmin(distance_matrix)
        return min_index

    def _find_k_similar(self, data_point, cd, k):
        # find euclidean distance between data point and entire cd
        distance_matrix = np.linalg.norm(np.array(cd) - data_point, axis=1)
        # Manhattan distance
        # distance_matrix = distance.cityblock(np.array(cd),data_point)
        # canberra distance
        # distance_matrix = distance.canberra(np.array(cd), data_point)
        # canberra distance
        # distance_matrix = distance.canberra(np.array(cd), data_point)

        self.distance_matrix_sorted = np.sort(distance_matrix[:k])
        # find the indexes of the most similar ones
        min_indices = np.argsort(distance_matrix)[:k]


        return min_indices

    def most_voted(self, min_indices):
        labels = []
        votes_ind = []
        # load sorted distances (data_point from cd)
        distances = self.distance_matrix_sorted
        mean_distances = []
        votes_dup = []
        # Append the labels sorting by min indices (from min distance to max distance)
        for m in min_indices:
            labels.append(self.cd_classes[m])
        # Retrieve the indices of the labels
        for l in set(labels):
            votes_ind.append([i for i, x in enumerate(labels) if x == l])
        # Calculate the most occurences from the labels
        votes_len = [len(i) for i in votes_ind]
        # check if there are ties
        for v in votes_ind:
            if len(v) == max(votes_len):
                distance = []
                for iv in v:
                    distance.append(distances[iv])
                mean_distances.append(np.mean(distance))
                votes_dup.append(v)
        # resolve the ties by precedence of the labels with the lowest mean distances
        where_min = np.where(mean_distances == min(mean_distances))
        most_voted_label = votes_dup[where_min[0][0]]
        return most_voted_label

    def plurality_voted(self, min_indices):
        labels = []
        # load sorted distances (data_point from cd)
        distances = self.distance_matrix_sorted
        # Append the labels sorting by min indices (from min distance to max distance)
        for m in min_indices:
            labels.append(self.cd_classes[m])
        k = len(distances)
        for kk in range(k):
            votes_ind = []
            labels = labels[:k]
            # Retrieve the indices of the labels
            for l in set(labels):
                votes_ind.append([i for i, x in enumerate(labels) if x == l])
            # Calculate the most occurences from the labels
            votes_len = [len(i) for i in votes_ind]
            duplicate = []
            # Check if there are duplicates
            for v in votes_ind:
                if len(v) == max(votes_len):
                    duplicate.append(1)
            # If there are duplicates, reduce the number of k
            if len(duplicate) > 1:
                k = k - 1
            else:
                # If there are no ties, append the label with most votes
                where = votes_len.index(max(votes_len))
                final_index = votes_ind[where][0]
                return labels[final_index]

    def borda_voted(self, min_indices):
        labels = []
        # load sorted distances (data_point from cd)
        distances = self.distance_matrix_sorted
        # Append the labels sorting by min indices (from min distance to max distance)
        for m in min_indices:
            labels.append(self.cd_classes[m])
        k = len(distances)
        for i in range(k):
            duplicate = []
            scores = []
            scores_sum = []
            labels = labels[:k]
            lbls = list(set(labels))
            # Create a scoring list
            for kk in range(k):
                score = k - (kk + 1)
                scores.append(score)
            scores = np.array(scores)
            # For every vote append the score by assigning k - 1 to the best, and 0 to the worst
            for l in lbls:
                ind = np.where(labels == l)
                scores_sum.append(np.sum(scores[ind]))
            for s in scores_sum:
                # Check if there are ties
                if s == max(scores_sum):
                    duplicate = np.append(duplicate, 1)
            # If there are ties reduce the k number
            if len(duplicate) > 1:
                k = k - 1
            # If there are no ties, append the label with the highest score
            else:
                where = np.where(max(scores_sum))
                final_label = lbls[where[0][0]]
                return final_label



    # calculate accuracy
    def calculate_accuracy(self):
        return self.correct_num / self.dataset_size

    def print_results(self):
        print(f"accuracy: {self.calculate_accuracy() * 100} %")
        print(f"execution time: {self.time}")
        print(f"used memory: {len(self.cd)} / {self.total_size}")
