import time

from ib.IB import IB
from utils.arff_parser import arff_to_df_normalized

if __name__ == '__main__':
    tic = time.perf_counter()
    # data = 'datasets/vehicle.arff'
    data = 'datasetsCBR/bal/bal.fold.000000.train.arff'

    df_normalized, data_names, classes = arff_to_df_normalized(data)

    df_numpy = df_normalized.to_numpy()

    ib_class = IB('ib2')
    ib_class.fit_and_predict(df_numpy, classes)
    ib_class.print_results()
