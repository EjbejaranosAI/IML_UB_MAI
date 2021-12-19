from model_evaluator.model_evaluator import ModelEvaluator
from utils.validation_tests import t_test
from scipy import stats

if __name__ == '__main__':
    model_evaluator = ModelEvaluator()
    model_evaluator.evaluate_model(algorithm='ib1', dataset_name='pen-based')
    model_evaluator.evaluate_model(algorithm='ib2', dataset_name='pen-based')
    model_evaluator.evaluate_model(algorithm='ib3', dataset_name='pen-based')
    print(model_evaluator.perfomance)
    t_test(model_evaluator.perfomance)

