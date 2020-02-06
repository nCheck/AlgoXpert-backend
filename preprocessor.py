import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def preprocess( FILE , TARGET , UNWANTED ):

    FILE = 'uploads/' + FILE
    

    f = open(FILE , 'r+')

    HEADER , SEP = findHeaderAndSEP(f)

    f.close()

    # Open File

    data = pd.read_csv(FILE , sep=SEP , header=HEADER )

    #Remove Unwanted
            
    data = data.drop(UNWANTED , axis=1)

    # Keep Data with Finite Target

    data = data[ np.isfinite( data[TARGET] ) ]

    #Remove more than half missing data

    data = data.dropna(thresh=0.5,axis=1)

    #Seperating X and Y

    if TARGET is not None:
        Y = data[TARGET]
        X = data.drop([TARGET] , axis = 1)
        data = X

    # One Hot Encode String data

    def encode_and_bind(original_dataframe, feature_to_encode):
        dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
        res = pd.concat([original_dataframe, dummies], axis=1)
        res = res.drop([feature_to_encode] , axis=1)
        return res

    columns = data.columns

    for c in columns:
        
        if data[c].dtype == 'object':
            
            data = encode_and_bind(data , c)

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    data = imputer.fit_transform(data)

    if TARGET is None:
        return data

    # Train-Test Split

    x_train,x_test,y_train,y_test = train_test_split( data ,Y,test_size=.34)

    return x_train,x_test,y_train,y_test



def findHeaderAndSEP(f):

    # Finds Seperator

    line = f.readline().strip()
    SEP = None
    if ',' in line:
        SEP = ','
    elif ':' in line:
        SEP = ':'
    elif ';' in line:
        SEP = ';'

    line1 = line.split(SEP)
    line2 = f.readline().strip().split(SEP)

    if SEP is None:
        SEP = '\s+'

    types1 = []
    types2 = []

    # Finds Header

    for l in line1:
        try:
            float(l)
            types1.append('float')
        except:
            types1.append('str')
            
            
    for l in line2:
        try:
            float(l)
            types2.append('float')
        except:
            types2.append('str')
        
        
    HEADER = None
    for a , b in zip( types1 , types2 ):
        if a != b:
            HEADER = 0
            break
    
    # print("HEADER, SEP", HEADER , SEP)
    
    return HEADER , SEP






# preprocess('uploads/data.txt', 'quality' , [])