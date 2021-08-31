import argparse
import logging

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import math

def load_data_MNTest(fl="C:/Users/Vineet/Desktop/mission future/Auto_ML/Week2_Eval_of_ML_Models/Assignment/ex02-evaluation/src/MCTestData.csv"):
    """
    Loads data stored in McNemarTest.csv
    :param fl: filename of csv file
    :return: labels, prediction1, prediction2
    """
    data = pd.read_csv(fl, header=None).to_numpy()
    labels = data[:, 0]
    prediction_1 = data[:, 1]
    prediction_2 = data[:, 2]
    return labels, prediction_1, prediction_2


def load_data_TMStTest(fl="C:/Users/Vineet/Desktop/mission future/Auto_ML/Week2_Eval_of_ML_Models/Assignment/ex02-evaluation/src/TMStTestData.csv"):
    """
    Loads data stored in fl
    :param fl: filename of csv file
    :return: y1, y2
    """
    data = np.loadtxt(fl, delimiter=",")
    y1 = data[:, 0]
    y2 = data[:, 1]
    return y1, y2


def load_data_FTest(fl="C:/Users/Vineet/Desktop/mission future/Auto_ML/Week2_Eval_of_ML_Models/Assignment/ex02-evaluation/src/FTestData.csv"):
    """
    Loads data stored in fl
    :param fl: filename of csv file
    :return: evaluations
    """
    errors = np.loadtxt(fl, delimiter=",")
    return errors


def McNemar_test(labels, prediction_1, prediction_2):
    """
    TODO
    :param labels: the ground truth labels
    :param prediction_1: the prediction results from model 1
    :param prediction_2:  the prediction results from model 2
    :return: the test statistic chi2_Mc
    """
    #Hypothesis
    #HO:Two models performance are similar 
    #H1:Two models performance are not similar
    
    leng=len(labels)
    A=0
    B=0
    C=0
    D=0
    #Create Confusion Matrix
    for i in range(leng):
        if (labels[i] == prediction_1[i] and labels[i] == prediction_2[i]):
            A=A+1
        elif (labels[i] == prediction_1[i] and labels[i] != prediction_2[i]):
            B=B+1
        elif (labels[i] != prediction_1[i] and labels[i] == prediction_2[i]):
            C=C+1
        else:
            D=D+1 

    #print(A,B,C,D,A+B+C+D)

    #Condition to check whether to use contuinity equation 
    if(B+C<=20):
        print("McNemar test must not be used")
        return()
    #Calculate Chi_2_MC
    chi2_Mc=((abs(B-C)-1)**2)/(B+C)
    print(chi2_Mc)
    #chi2 = np.random.uniform(0, 1)
    #print(chi2)
    return chi2_Mc


def TwoMatchedSamplest_test(y1, y2):
    """
    TODO
    :param y1: runs of algorithm 1
    :param y2: runs of algorithm 2
    :return: the test statistic t-value
    """
    #y3: diff between two algoritm's outer loss
    #dmean: Mean of y3
    #sd:Standard Deviation of y3

    y3=[]
    n_test=len(y1)
    for i in range(n_test):
        y3.append(y1[i]-y2[i])
        
    #Calculating Mean
    dmean=np.mean(y3)

    #Calculating Standard Deviation
    sd=np.std(y3)

    #Calculating t_value
    t_value=math.sqrt(n_test)*(dmean/sd)
    #print(t_value)

    #t_value = np.random.uniform(0, 1)
    return t_value


def Friedman_test(errors):
    """
    TODO
    :param errors: the error values of different algorithms on different datasets
    :return: chi2_F: the test statistic chi2_F value
    :return: FData_stats: the statistical data of the Friedan test data, you can add anything needed to facilitate
    solving the following post hoc problems
    """
    n=15
    k=5
    SS_Total=0
    SS_Error=0
    #print(errors)

    #Constructing Rank Matrix
    rank_mat=np.array([[sorted(set(i)).index(x) + 1 for x in i] for i in errors])
    R_bar=np.sum(rank_mat)/(n*k)
    for j in range(k):
        R_bar_j=np.mean(rank_mat[:,j])
        SS_Total=SS_Total+(R_bar-R_bar_j)**2
    SS_Total=SS_Total*n
    for i in range(n):
        for j in range(k):
            SS_Error=SS_Error+(rank_mat[i][j]-R_bar)**2
    SS_Error=SS_Error/(n*(k-1))
    chi2_F=SS_Total/SS_Error
    print(chi2_F)
    #print(rank_mat)
    #chi2_F = np.random.uniform(0, 1)
    FData_stats = {'errors': errors,'rank':rank_mat}
    return chi2_F, FData_stats


def Nemenyi_test(FData_stats):
    """
    TODO
    :param FData_stats: the statistical data of the Friedan test data to be utilized in the post hoc problems
    :return: the test statisic Q value
    """
    k=5 
    n=15
    Q_value=np.zeros((5,5))
    rank_lst=[]
    rank_mat=FData_stats['rank']
    div=math.sqrt((k*(k+1))/(6*n))   

    #Calculating avg Rank for all algorithms throught all datasets
    for j in range(k):
        rank_lst.append(np.mean(rank_mat[:,j]))

    
   #Constructing Q_Value Matrix
    for i in range(k):
        for j in range(i+1,k):
            Q_value[i][j]=(rank_lst[i]-rank_lst[j])/div

    #print(Q_value)
    #Q_value = np.empty_like([1])
    return Q_value


def box_plot(FData_stats):
    """
    TODO
    :param FData_stats: the statistical data of the Friedan test data to be utilized in the post hoc problems
    """
    error_mat=FData_stats['errors']
    rank_mat=FData_stats['rank']
    k=5
    data=[]
    rank_lst=[]
    for j in range(k):
        rank_lst.append(np.mean(rank_mat[:,j]))
    
    Best_avg_rank_index=rank_lst.index(np.max(rank_lst))
    Worst_avg_rank_index=rank_lst.index(np.min(rank_lst))

   
    data.append(error_mat[:,Best_avg_rank_index])
    data.append(error_mat[:,Worst_avg_rank_index])
    labels=['A1','A2']

    fig1 , ax1=plt.subplots()
    a=ax1.boxplot(data,patch_artist=True,labels=labels)
   



    # fill with colors
    colors = ['lightblue', 'lightgreen']
    
    for patch, color in zip(a['boxes'], colors):
         patch.set_facecolor(color)
    ax1.legend([a ['boxes'][0],a ['boxes'][0]] ,['A1','A2'],loc='upper right') 

    ax1.set_xlabel('Worst and Best Ranked Algorithms')
    ax1.set_ylabel('Outer Loss')
    
    plt.show()
    pass


def main(args):
    # (a)
    labels, prediction_A, prediction_B = load_data_MNTest()
    chi2_Mc = McNemar_test(labels, prediction_A, prediction_B)

    # (b)
    y1, y2 = load_data_TMStTest()
    t_value = TwoMatchedSamplest_test(y1, y2)

    # (c)
    errors = load_data_FTest()
    chi2_F, FData_stats = Friedman_test(errors)

    # (d)
    Q_value = Nemenyi_test(FData_stats)

    # (e)
    box_plot(FData_stats)


if __name__ == '__main__':
    cmdline_parser = argparse.ArgumentParser('ex03')

    cmdline_parser.add_argument('-v', '--verbose', default='INFO', choices=['INFO', 'DEBUG'], help='verbosity')
    cmdline_parser.add_argument('--seed', default=12345, help='Which seed to use', required=False, type=int)
    args, unknowns = cmdline_parser.parse_known_args()
    np.random.seed(args.seed)
    log_lvl = logging.INFO if args.verbose == 'INFO' else logging.DEBUG
    logging.basicConfig(level=log_lvl)

    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')
    main(args)
