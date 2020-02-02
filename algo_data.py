import pandas as pd



def getStats( algos ):

    data = pd.read_csv('DataMiningAlgoAnalysis.csv', header=0)  
    data = data.to_dict('index')

    res = {}
    for d in data.keys():

        if data[d]['Name'] in algos:

            res[ data[d]['Name'] ] = data[d] 

    return res



def test():
    algos = ['ExtraTreeClassifier' , 'AdaBoostClassifier']
    print( getStats(algos) )


