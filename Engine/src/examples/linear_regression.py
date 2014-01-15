from algorithms.linearRegression.LinearRegressionFactory import LinearRegressionFactory
from datahandler.numerical.NumericalDataHandler import NumericalDataHandler
from engine.Engine import Engine


if __name__ == '__main__':
    
    nrParams = 11
    nrLabelDim = 1
    
    factory = LinearRegressionFactory(nrParams)
    data_handler = NumericalDataHandler(nrParams, nrLabelDim) 
    engine = Engine()
#     engine.init(factory, data_handler)
    
    engine.run()