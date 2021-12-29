from model_evaluator.model_evaluator import ModelEvaluator
from model_evaluator.model_evaluator import ModelEvaluatorFeatureSelection
if __name__ == '__main__':
    #model_evaluator = ModelEvaluator(dataset='satimage')
    model_evaluator = ModelEvaluatorFeatureSelection(dataset='satimage')

     #model_evaluator = ModelEvaluator(dataset='hypothyroid')

    # performing evaluation on ibs
     #model_evaluator.evaluate_model(algorithm='ib1')
     #model_evaluator.evaluate_model(algorithm='ib2')
     #model_evaluator.evaluate_model(algorithm='ib3')

    # finding best one
    # best_algorithm = model_evaluator.find_best_ib()

    # model_evaluator.find_best_configuration(algorithm=best_algorithm)
    # print(model_evaluator.k_perfomance)

    #print(model_evaluator.dataset[5]['train_data'])

