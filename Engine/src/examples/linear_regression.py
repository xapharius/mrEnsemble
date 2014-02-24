from algorithms.linearRegression.LinearRegressionFactory import LinearRegressionFactory
from datahandler.numerical.NumericalDataHandler import NumericalDataHandler
from engine.engine import Engine


if __name__ == '__main__':
    
    # 1. define algorithm
    nrParams = 11
    regression = LinearRegressionFactory(nrParams)
    
    # 2. set data handler
    nrLabelDim = 1
    data_handler = NumericalDataHandler(nrParams, nrLabelDim)
    
    # 3. run
    engine = Engine(regression, data_handler, 'hdfs:///user/linda/ml/data/winequality-red.csv')
    engine.start()