from ib.ib import IB
from utils.arff_parser import arff_to_df_normalized


class ModelEvaluator:
    def __init__(self):
        self.dataset_path = 'datasetsCBR'
        self.accuracy = []
        self.time = 0
        self.perfomance = {}

    def evaluate_model(self, algorithm, dataset_name):
        self.accuracy = []
        self.memories = []
        self.time = 0
        self.perfomance[algorithm] = {}

        for fold in range(10):
            model = IB(algorithm)

            # load data
            train_data, train_labels = self._load_dataset(dataset_name, fold, 'train')
            test_data, test_labels = self._load_dataset(dataset_name, fold, 'test')

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

    # load normalized data
    def _load_dataset(self, dataset_name, fold, mode):
        dataset_path = f"{self.dataset_path}/{dataset_name}/{dataset_name}.fold.{fold:06}.{mode}.arff"
        data, labels, _ = arff_to_df_normalized(dataset_path)

        return data.to_numpy(), labels

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
