from ib.ib import IB
from utils.arff_parser import arff_to_df_normalized


class ModelEvaluator:
    def __init__(self):
        self.dataset_path = 'datasetsCBR'
        self.accuracy = []
        self.time = 0

    def evaluate_model(self, algorithm, dataset_name):
        self.accuracy = []
        self.time = 0

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

            # save evaluation time
            self.time += model.time

            # print results
            print(f"\n\nModel:{algorithm} \tFold:{fold} \n")
            model.print_results()

        # print evaluation results
        self._print_evaluation_results()

    # load normalized data
    def _load_dataset(self, dataset_name, fold, mode):
        dataset_path = f"{self.dataset_path}/{dataset_name}/{dataset_name}.fold.{fold:06}.{mode}.arff"
        data, labels, _ = arff_to_df_normalized(dataset_path)

        return data.to_numpy(), labels

    # print evaluation results
    def _print_evaluation_results(self):
        print('\n\n\n')
        print(f'Mean accuracy of the model is: {sum(self.accuracy) * 100 / len(self.accuracy)}%')
        print(f'Execution time for evaluation is: {self.time}')
