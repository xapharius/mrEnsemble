from algorithms.linearRegression.LinearRegressionFactory import LinearRegressionFactory
from datahandler.numerical.NumericalDataHandler import NumericalDataHandler
from engine.engine import Engine


if __name__ == '__main__':
    
    nrParams = 11
    nrLabelDim = 1
    
    # 1. define algorithm
    regression = LinearRegressionFactory(nrParams)
    
    # 2. set data handler
    data_handler = NumericalDataHandler(nrParams, nrLabelDim)
    
    # 3. run
    engine = Engine(regression, data_handler, 'hdfs:///user/linda/ml/data/winequality-red.csv')
    trained_alg = engine.start()
    
    # 4. do something good
