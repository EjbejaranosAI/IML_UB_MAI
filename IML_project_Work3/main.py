from model_evaluator.model_evaluator import ModelEvaluator

if __name__ == '__main__':
    model_evaluator = ModelEvaluator()
    model_evaluator.evaluate_model(algorithm='ib1', dataset_name='bal')
    model_evaluator.evaluate_model(algorithm='ib2', dataset_name='bal')
    model_evaluator.evaluate_model(algorithm='ib3', dataset_name='bal')
