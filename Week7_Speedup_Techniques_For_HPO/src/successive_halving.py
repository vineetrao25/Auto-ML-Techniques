import numpy as np

import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

from src.data.fcnet_benchmark import FCNetProteinStructureBenchmark  # noqa: E402
from src.utils import plot_grey_box_optimization  # noqa: E402


def successive_halving(problem, n_models, min_budget_per_model, max_budget_per_model, eta, random_seed):
    """
    The successive halving algorithm, called as subroutine in hyperband.
    :param problem: An instance of problem
    :param n_models: int;  The number of configs to evaluate
    :param min_budget_per_model: int
    :param max_budget_per_model: int
    :param eta: float
    :param random_seed: int
    :return:
    """
    np.random.seed(random_seed)
    configs_dict = {i: {'config': problem.get_configuration_space().sample_configuration(),
                        'f_evals': {}} for i in range(n_models)}
    #print("debug1",configs_dict)
    configs_to_eval = list(range(n_models))
    #print("debug2",configs_to_eval)
    b = np.int(min_budget_per_model)
    #print("debug3",b)
    while b <= max_budget_per_model:
        # Evaluate the configs selected for this budget
        for config_id in configs_to_eval:
            configs_dict[config_id]['f_evals'][b] = problem.objective_function(configs_dict[config_id]['config'],
                                                                               budget=b)
        #print("here1")                                                                   
        #rint(configs_dict[0]['config'])
        # Todo: Compute number of configs to proceed to next higher budget
        num_configs_to_proceed = int(len(configs_to_eval)/eta)
        #raise NotImplementedError()

        # Todo: Select the configs from the configs_dict which have been evaluated on the current budget b
        eval_configs_curr_budget = []
        tmp=[]
        for config_id in configs_to_eval:
            eval_configs_curr_budget.append(configs_dict[config_id]['config'])
            tmp.append((config_id,configs_dict[config_id]['f_evals'][b][0]))
        #print("here2")
        #print(eval_configs_curr_budget)
        #print(len(eval_configs_curr_budget))
        #print(tmp)
        tmp1=sorted(tmp,key=lambda x:x[1],reverse=False)
        #print(tmp1)
        #raise NotImplementedError()

        # Todo: Out of these configs select the ones to proceed to the next higher budget and assign this
        # list to configs_to_eval
        tmp2 = tmp1[:num_configs_to_proceed]
        configs_to_eval=[t[0] for t in tmp2]
        #print(configs_to_eval)
        best_config_id_in_this_iter=[]
        best_config_id_in_this_iter.append(configs_to_eval[0])
         
        #raise NotImplementedError()

        # Todo: Increase the budget for the next SH iteration.
        b =b*eta
        #print(b)

        new_dict={k:configs_dict[k] for k in configs_to_eval}
        best_config={j:configs_dict[j] for j in best_config_id_in_this_iter}
        if(len(configs_to_eval)==1):
            break
        #print("here",best_config)
        #print()
        #raise NotImplementedError()
        #print(configs_dict)

        
            
    #print(new_dict)    
    return best_config


if __name__ == '__main__':
    problem = FCNetProteinStructureBenchmark(data_dir="C:/Users/Vineet/Desktop/mission future/Auto_ML/Week7_Speedup_Techniques_For_HPO/Assignment/ex07-grey-box/src/data/fcnet_tabular_benchmarks/fcnet_tabular_benchmarks/")
    configs_dict = successive_halving(problem=problem, n_models=40, eta=2, random_seed=0, max_budget_per_model=100,
                                      min_budget_per_model=10)
    print(len(list(configs_dict)))
    plot_grey_box_optimization([configs_dict], min_budget_per_model=10)
