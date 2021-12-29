import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize, StandardScaler

from ib.ib import IB
from ib.voting_policies import most_voted, plurality_voted, borda_voted
from utils.arff_parser import arff_to_df_normalized
from utils.featureSelection import selectionIBLAlgorithm
from utils.validation_tests import rank_data, nemenyi_test


class ModelEvaluator:
    def __init__(self, dataset):
        self.dataset_path = 'datasetsCBR'
        self.time = 0
        self.perfomance = {}
        self.k_perfomance = {}
        self.dataset = {}
        self.number_of_folds = 10

        self.configuration_matrices = {}
        self.configuration_mapping = {}

        self.dataset_name = dataset

        # options
        self.k_options = [1, 3, 5, 7]
        self.distance_algorithms = {'HVDM': 'HVDM', 'Eucledian': None, 'cityblock': 'cityblock', 'canberra': 'canberra'}
        self.voting_policies = {'borda_voted': borda_voted, 'most_voted': most_voted, 'plurality_voted': plurality_voted}

        # load folds
        for fold in range(self.number_of_folds):
            # load data
            self.dataset[fold] = {}

            train_data, train_labels, column_names = self._load_dataset(dataset, fold, 'train')
            test_data, test_labels, _ = self._load_dataset(dataset, fold, 'test')

            train_data, test_data = self._preprocess_dataset(train_data, test_data, column_names)

            self.dataset[fold]['train_data'] = train_data
            self.dataset[fold]['train_labels'] = train_labels
            self.dataset[fold]['test_data'] = test_data
            self.dataset[fold]['test_labels'] = test_labels

            print(f'Fold {fold + 1} Loaded')

    def evaluate_model(self, algorithm):
        accuracy = []
        memories = []
        self.time = 0
        self.perfomance[algorithm] = {}

        for fold in range(self.number_of_folds):
            model = IB(algorithm)

            # load data
            train_data, train_labels = self.dataset[fold]['train_data'], self.dataset[fold]['train_labels']
            test_data, test_labels = self.dataset[fold]['test_data'], self.dataset[fold]['test_labels']

            # fit and predict data
            model.fit_and_predict(train_data, train_labels)
            model.fit_and_predict(test_data, test_labels)

            # calculate test accuracy and save
            accuracy.append(model.calculate_accuracy())

            # get memories used
            memories.append(model.get_memory())

            # save evaluation time
            self.time += model.time

            # print results
            print(f"\n\nAlgorithm:{algorithm} \tFold:{fold + 1} \n")
            model.print_results()

        self.perfomance[algorithm]['accuracy'] = self._calculate_mean(accuracy)
        self.perfomance[algorithm]['variance'] = self._calculate_variance(accuracy)
        self.perfomance[algorithm]['time'] = self.time
        self.perfomance[algorithm]['memory'] = self._calculate_mean(memories)

        # print evaluation results
        self._print_evaluation_results(algorithm, accuracy)

    def find_best_configuration(self, algorithm):
        self.accuracy = []
        self.memories = []
        self.configuration_matrices = {}
        self.configuration_mapping = {}

        self.time = 0

        inx = 1
        for k in self.k_options:
            for distance_alg in self.distance_algorithms.keys():
                for voting_policy in self.voting_policies.keys():
                    self.k_perfomance[inx] = {}

                    self.k_perfomance[inx]['options'] = {}
                    self.k_perfomance[inx]['result'] = {}
                    accuracy = []
                    memories = []
                    time = 0
                    for fold in range(self.number_of_folds):
                        model = IB(algorithm)

                        # load data
                        train_data, train_labels = self.dataset[fold]['train_data'], self.dataset[fold]['train_labels']
                        test_data, test_labels = self.dataset[fold]['test_data'], self.dataset[fold]['test_labels']

                        # fit and predict data
                        model.fit_and_predict(train_data, train_labels, k, self.distance_algorithms[distance_alg],
                                              self.voting_policies[voting_policy])
                        model.fit_and_predict(test_data, test_labels, k, self.distance_algorithms[distance_alg],
                                              self.voting_policies[voting_policy])

                        # calculate test accuracy and save
                        accuracy.append(model.calculate_accuracy())

                        # get memories used
                        memories.append(model.get_memory())

                        # save evaluation time
                        time += model.time

                        # print results
                        # print(f"\n\nModel:{algorithm} \tFold:{fold} \n")
                        # model.print_results()

                    self.configuration_mapping[inx] = {'k': k, 'distance_algorithm': distance_alg,
                                                       'voting_policy': voting_policy}
                    self.configuration_matrices[inx] = accuracy

                    self.k_perfomance[inx]['options'] = {'k': k, 'distance_algorithm': distance_alg,
                                                         'voting_policy': voting_policy}
                    self.k_perfomance[inx]['result']['accuracy'] = self._calculate_mean(accuracy)
                    self.k_perfomance[inx]['result']['variance'] = self._calculate_variance(accuracy)
                    self.k_perfomance[inx]['result']['time'] = time
                    self.k_perfomance[inx]['result']['memory'] = self._calculate_mean(memories)
                    print(f'following configurations executed: {self.k_perfomance[inx]["options"]}, '
                          f'mean accuracy: {self.k_perfomance[inx]["result"]["accuracy"]}')
                    inx += 1
        ranked_data = rank_data(self.configuration_matrices)
        nemenyi = nemenyi_test(self.configuration_matrices)

    def find_best_ib(self):
        scores = {}
        for algorithm in self.perfomance.keys():
            result = self.perfomance[algorithm]

            scores[algorithm] = result['accuracy'] * 0.9 - result['variance'] * 0.24 * 100 + \
                                (1 / (result['memory'])) * 0.001 + (1 / (result['time'])) * 0.001
        best_algorithm = max(scores, key=lambda x: scores[x])
        print(f'\n\n\nBest Algorithm for "{self.dataset_name}" dataset is: {best_algorithm}')
        return max(scores, key=lambda x: scores[x])

    # load normalized data
    def _load_dataset(self, dataset_name, fold, mode):
        dataset_path = f"{self.dataset_path}/{dataset_name}/{dataset_name}.fold.{fold:06}.{mode}.arff"
        data, labels, column_names = arff_to_df_normalized(dataset_path)
        labels = [label.decode("utf-8") for label in labels.to_numpy()]
        return data, np.array(labels), column_names

    def _preprocess_dataset(self, train_data, test_data, column_names):
        # remove columns that has more than 80% nans
        train_data = train_data.dropna(thresh=len(train_data) * 0.8, axis=1)
        train_data = train_data.fillna(train_data.mean())

        data_names = [name for name in column_names if name in train_data.columns]

        # scale and normalize
        scalar = StandardScaler()
        df_scaled = scalar.fit_transform(train_data)
        df_normalized = normalize(df_scaled)
        train_data = pd.DataFrame(df_normalized, columns=data_names)

        test_data = test_data[data_names]
        test_data = test_data.fillna(test_data.mean())

        scalar = StandardScaler()
        df_scaled = scalar.fit_transform(test_data)
        df_normalized = normalize(df_scaled)
        test_data = pd.DataFrame(df_normalized, columns=data_names)

        return train_data.to_numpy(), test_data.to_numpy()

    # print evaluation results
    def _print_evaluation_results(self, algorithm, accuracy):
        print('\n\n\n')
        print(f'Final Results of the {algorithm} Algorithm:')
        print(f'Mean accuracy of the model is: {sum(accuracy) * 100 / len(accuracy)}%')
        print(f'Execution time for evaluation is: {self.time}')

    def _calculate_mean(self, array):
        return sum(array) / len(array)

    def _calculate_variance(self, array):
        mean_value = self._calculate_mean(array)
        variance = sum([(val - mean_value) ** 2 for val in array]) / (len(array) - 1)
        return variance

    def select_features(self, method, number_of_features):
        data = self.dataset[0]['train_data']
        labels = self.dataset[0]['train_labels']
        feature_indices = selectionIBLAlgorithm(data, labels, method, number_of_features)

        for fold in range(self.number_of_folds):
            # load data

            self.dataset[fold]['train_data_fs'] = self.dataset[fold]['train_data'][:, feature_indices]
            self.dataset[fold]['test_data_fs'] = self.dataset[fold]['test_data'][:, feature_indices]

            print(f'Created new data with new Features for Fold: {fold + 1}')

    def k_ibl(self, algorithm, k, voting_policy, distance_alg, feature_selection):
        accuracy = []
        memories = []
        time = 0

        if feature_selection:
            train_data_property_name = 'train_data_fs'
            test_data_property_name = 'test_data_fs'
        else:
            train_data_property_name = 'train_data'
            test_data_property_name = 'test_data'

        for fold in range(self.number_of_folds):
            model = IB(algorithm)

            # load data
            train_data, train_labels = self.dataset[fold][train_data_property_name], self.dataset[fold]['train_labels']
            test_data, test_labels = self.dataset[fold][test_data_property_name], self.dataset[fold]['test_labels']

            # fit and predict data
            model.fit_and_predict(train_data, train_labels, k, self.distance_algorithms[distance_alg],
                                  self.voting_policies[voting_policy])
            model.fit_and_predict(test_data, test_labels, k, self.distance_algorithms[distance_alg],
                                  self.voting_policies[voting_policy])

            # calculate test accuracy and save
            accuracy.append(model.calculate_accuracy())

            # get memories used
            memories.append(model.get_memory())

            # save evaluation time
            time += model.time

        mean_accuracy = self._calculate_mean(accuracy)
        mean_memory = self._calculate_mean(memories)

        print(f'following configurations executed: k: {k}, voting policy: {voting_policy}, distance algorithm: {distance_alg} '
              f'mean accuracy: {mean_accuracy * 100}%, memory: {mean_memory * 100}%, time used: {time}, feature selection: {feature_selection}')
