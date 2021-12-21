from sklearn.preprocessing import normalize, StandardScaler
import pandas as pd

from ib.ib import IB
from ib.voting_policies import most_voted, plurality_voted, borda_voted
from utils.arff_parser import arff_to_df_normalized


class ModelEvaluator:
    def __init__(self, dataset):
        self.dataset_path = 'datasetsCBR'
        self.accuracy = []
        self.time = 0
        self.perfomance = {}
        self.k_perfomance = {}
        self.dataset = {}
        self.number_of_folds = 10
        for fold in range(self.number_of_folds):
            # load data
            self.dataset[fold] = {}

            train_data, train_labels, _ = self._load_dataset(dataset, fold, 'train')
            test_data, test_labels, column_names = self._load_dataset(dataset, fold, 'test')

            train_data, test_data = self._preprocess_dataset(train_data, test_data, column_names)

            self.dataset[fold]['train_data'] = train_data
            self.dataset[fold]['train_labels'] = train_labels
            self.dataset[fold]['test_data'] = test_data
            self.dataset[fold]['test_labels'] = test_labels

            print(f'Fold {fold + 1} Loaded')

    def evaluate_model(self, algorithm):
        self.accuracy = []
        self.memories = []
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
            self.accuracy.append(model.calculate_accuracy())

            # get memories used
            self.memories.append(model.get_memory())

            # save evaluation time
            self.time += model.time

            # print results
            print(f"\n\nModel:{algorithm} \tFold:{fold} \n")
            model.print_results()

        self.perfomance[algorithm]['accuracy'] = self._calculate_mean(self.accuracy)
        self.perfomance[algorithm]['variance'] = self._calculate_variance(self.accuracy)
        self.perfomance[algorithm]['time'] = self.time
        self.perfomance[algorithm]['memory'] = self._calculate_mean(self.memories)

        # print evaluation results
        self._print_evaluation_results(algorithm)

    def find_best_configuration(self, algorithm):
        self.accuracy = []
        self.memories = []
        self.time = 0

        k_options = [1, 3, 5, 7]
        distance_algorithms = {'HVDM': 'HVDM', 'Eucledian': None, 'cityblock': 'cityblock', 'canberra': 'canberra'}
        voting_policies = {'borda_voted': borda_voted, 'most_voted': most_voted, 'plurality_voted': plurality_voted}

        inx = 1
        for k in k_options:
            for distance_alg in distance_algorithms.keys():
                for voting_policy in voting_policies.keys():
                    self.k_perfomance[inx] = {}

                    self.k_perfomance[inx]['options'] = {}
                    self.k_perfomance[inx]['result'] = {}
                    accuracy = []
                    memories = []
                    time = 0
                    for fold in range(10):
                        model = IB(algorithm)

                        # load data
                        train_data, train_labels = self.dataset[fold]['train_data'], self.dataset[fold]['train_labels']
                        test_data, test_labels = self.dataset[fold]['test_data'], self.dataset[fold]['test_labels']

                        # fit and predict data
                        model.fit_and_predict(train_data, train_labels, k, distance_algorithms[distance_alg], voting_policies[voting_policy])
                        model.fit_and_predict(test_data, test_labels, k, distance_algorithms[distance_alg], voting_policies[voting_policy])

                        # calculate test accuracy and save
                        accuracy.append(model.calculate_accuracy())

                        # get memories used
                        memories.append(model.get_memory())

                        # save evaluation time
                        time += model.time

                        # print results
                        # print(f"\n\nModel:{algorithm} \tFold:{fold} \n")
                        # model.print_results()

                    self.k_perfomance[inx]['options'] = {'k': k, 'distance_algorithm': distance_alg, 'voting_policy': voting_policy}
                    self.k_perfomance[inx]['result']['accuracy'] = self._calculate_mean(accuracy)
                    self.k_perfomance[inx]['result']['variance'] = self._calculate_variance(accuracy)
                    self.k_perfomance[inx]['result']['time'] = time
                    self.k_perfomance[inx]['result']['memory'] = self._calculate_mean(memories)
                    print(f'following configurations executed: {self.k_perfomance[inx]["options"]}, '
                          f'mean accuracy: {self.k_perfomance[inx]["result"]["accuracy"]}')
                    inx += 1
                # print(self.k_perfomance)

    # load normalized data
    def _load_dataset(self, dataset_name, fold, mode):
        dataset_path = f"{self.dataset_path}/{dataset_name}/{dataset_name}.fold.{fold:06}.{mode}.arff"
        data, labels, column_names = arff_to_df_normalized(dataset_path)

        return data, labels, column_names

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
    def _print_evaluation_results(self, algorithm):
        print('\n\n\n')
        print(f'Final Results of the {algorithm} Algorithm:')
        print(f'Mean accuracy of the model is: {sum(self.accuracy) * 100 / len(self.accuracy)}%')
        print(f'Execution time for evaluation is: {self.time}')

    def _calculate_mean(self, array):
        return sum(array) / len(array)

    def _calculate_variance(self, array):
        mean_value = self._calculate_mean(array)
        variance = sum([(val - mean_value) ** 2 for val in array]) / (len(array) - 1)
        return variance
