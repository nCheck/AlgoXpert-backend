from sklearn import tree
import numpy as np
from sklearn import ensemble
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score,mean_squared_error


modelPack = {}

def trees( x_train, x_test, y_train, y_test ):

    res = []

    m = tree.DecisionTreeRegressor()
    m.fit(x_train, y_train)

    predictions = m.predict(x_test)
    acc = mean_squared_error(y_test,predictions)

    modelPack['DecisionTreeRegressor'] = m

    res.append( ( acc , "DecisionTreeRegressor" ) )

    m = tree.ExtraTreeRegressor()
    m.fit(x_train, y_train)
    predictions = m.predict(x_test)
    acc = mean_squared_error(y_test,predictions)

    modelPack['ExtraTreeRegressor'] = m

    res.append( ( acc , "ExtraTreeRegressor" ) )

    print(res)

    return res


def ensembles( x_train, x_test, y_train, y_test ):

    res = []
    m = SVR(kernel='rbf', epsilon=.1)
    m.fit(x_train, y_train)
    predictions = m.predict(x_test)
    acc = mean_squared_error(y_test,predictions)

    modelPack['SVM-RBF'] = m

    res.append( ( acc , "SVM-RBF" ) )

    print("done RBF")

    m = SVR(kernel='poly', epsilon=.1)
    m.fit(x_train, y_train)
    predictions = m.predict(x_test)
    acc = mean_squared_error(y_test,predictions)


    modelPack["SVM-POLY"] = m

    res.append( ( acc , "SVM-POLY" ) )

    print("done POLY")





    return res



def lines( x_train, x_test, y_train, y_test ):

    res = []
    m = linear_model.Ridge(alpha=1.0)
    m.fit(x_train, y_train)
    predictions = m.predict(x_test)
    acc =mean_squared_error(y_test,predictions)

    modelPack["Ridge"] = m

    res.append( ( acc , "Ridge" ) )

    m = linear_model.LinearRegression()
    m.fit(x_train, y_train)
    predictions = m.predict(x_test)
    print("preds",predictions)
    acc =mean_squared_error(y_test,predictions)

    modelPack["Linear Regression"] = m

    res.append( ( acc , "Linear Regression" ) )

    m = linear_model.Lasso(alpha=0.1)
    m.fit(x_train, y_train)
    predictions = m.predict(x_test)
    acc = mean_squared_error(y_test,predictions)

    modelPack["Lasso"] = m

    res.append( ( acc , "Lasso" ) )

    m = linear_model.LassoLars(alpha=0.1)
    m.fit(x_train, y_train)
    predictions = m.predict(x_test)
    acc = mean_squared_error(y_test,predictions)

    modelPack["LassoLARs"] = m

    res.append( ( acc , "LassoLARs" ) )
    
    return res



def regression( x_train, x_test, y_train, y_test ):

    result = {}

    r1 = trees( x_train, x_test, y_train, y_test )
    r2 = lines( x_train, x_test, y_train, y_test )
    # r3 = ensembles( x_train, x_test, y_train, y_test )

    res = r1 + r2 

    res.sort()
    print(res)

    models = {}

    for val , name in res[:4]:
        result[name] = val
        models[name] = modelPack[name]


    return result , models