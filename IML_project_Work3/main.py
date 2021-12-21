from model_evaluator.model_evaluator import ModelEvaluator
# from utils.validation_tests import t_test
from scipy import stats

if __name__ == '__main__':
    # model_evaluator = ModelEvaluator(dataset='hypothyroid')
    model_evaluator = ModelEvaluator(dataset='bal')

    # model_evaluator.evaluate_model(algorithm='ib1')
    # model_evaluator.evaluate_model(algorithm='ib2')
    # model_evaluator.evaluate_model(algorithm='ib3')
    # print(model_evaluator.perfomance)
    # t_test(model_evaluator.perfomance)
    #
    model_evaluator.find_best_configuration(algorithm='ib1')
    print(model_evaluator.k_perfomance)

