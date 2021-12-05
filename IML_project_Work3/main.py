import time

from model_evaluator.model_evaluator import ModelEvaluator
from utils.arff_parser import arff_to_df_normalized

if __name__ == '__main__':
    model_evaluator = ModelEvaluator()
    model_evaluator.evaluate_model(algorithm='ib1', dataset_name='bal')
    model_evaluator.evaluate_model(algorithm='ib2', dataset_name='bal')
