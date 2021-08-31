import numpy as np
import logging
from evolution import Mutation, ParentSelection, Recombination, EA
from target_function import ackley
import itertools


def evaluate_black_box(mutation, selection, recombination):
    """
    Simple wrapper for your EA implementation. With your below hpo method you won't have to worry about
    other parameters
    :param mutation:
    :param selection:
    :param recombination:
    :return:
    """
    ea = EA(ackley, 20, 2, selection_type=selection,
            total_number_of_function_evaluations=500, problem_bounds=[-10, 10],
            mutation_type=mutation, recombination_type=recombination,
            sigma=1., children_per_step=5, fraction_mutation=.5, recom_proba=.5)
    print (ea.__dict__)
    #res = ea.optimize()
    #return res.fitness


def determine_best_hypers():
    """
    TODO implement either grid or random search to determine the best hyperparameter setting of your EA implementation
    when overfitting to the ackley function. The only parameter values you have to consider are selection_type,
    mutation_type and recombination type. You can treat the EA as a black-box by optimizing the black-box-function
    above. Note: the order of your "configuration" has to be as stated below

    :return: best configuration as tuple e.g. (mutation, selection, recombination) and performance value
    """
    # TODO
    mut=["Mutation.UNIFORM","Mutation.GAUSSIAN"]
    sel=["ParentSelection.NEUTRAL","ParentSelection.TOURNAMENT","Parent.Selection.FITNESS"]
    recomb=["Recombination.INTERMEDIATE","Recombination.UNIFORM"]
    comb=list(itertools.product(mut,sel,recomb))
    #print(comb)
    for i in comb:
        print(i[0],i[1],i[2])
        val=evaluate_black_box(i[0],i[1],i[2])

        print(val)

    #raise NotImplementedError
    return best_setting, best_perf
determine_best_hypers()
