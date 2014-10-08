'''
Created on Feb 4, 2014

@author: xapharius
'''
from algorithms.AbstractAlgorithm import AbstractAlgorithm
import numpy as np
import utils.numpyutils as nputils
from datahandler.numerical.NumericalDataSet import NumericalDataSet
from abc import abstractmethod, ABCMeta
from utils import logging


class AbstractUpdateMethod( object ):
    __metaclass__ = ABCMeta

    @abstractmethod
    def perform_update( self, weights, gradients, error ):
        '''
        @return: 
        '''
        pass


class SimpleUpdate( AbstractUpdateMethod ):

    def __init__( self, learning_rate ):
        '''
        @param learning_rate: Step width for gradient descent
        '''
        self.learning_rate = learning_rate

    def perform_update( self, weights, gradients, error ):
        for i in range( len( weights ) ):
            weights[ i ] -= self.learning_rate * gradients[ i ]
        return weights


class Rprop( AbstractUpdateMethod ):

    def __init__( self, layer_sizes, rate_pos=1.2, rate_neg=0.5,
                  init_step=0.0001, min_step=0.000001, max_step=50. ):
        self.rate_pos = rate_pos
        self.rate_neg = rate_neg
        self.max_step = max_step
        self.min_step = min_step
        self.last_gradient = [ ]
        self.step_size = [ ]
        # initialize last gradient and step size for each layer and weight
        for layer in range( len( layer_sizes ) - 1 ):
            init = np.zeros(
                (layer_sizes[ layer ] + 1, layer_sizes[ layer + 1 ]) )
            self.last_gradient.append( init )
            # set step size to given initial step size
            self.step_size.append( (init + 1) * init_step )

    def perform_update( self, weights, gradients, error ):
        for i in range( len( weights ) ):
            last_gradient = self.last_gradient[ i ]
            step_size = self.step_size[ i ]
            gradient = gradients[ i ]
            weight = weights[ i ]

            # calculate the change in the gradient direction
            change = np.sign( gradient * last_gradient )

            # get the weights where:
            # -  change > 0 -> direction didn't change
            #    -  change < 0 -> direction changed
            #    - change == 0 -> one of the gradients is 0, do nothing
            greater_zero_idxs = np.where( change > 0 )
            less_than_zero_idxs = np.where( change < 0 )

            # direction didn't change -> increase step size
            # ( probably we are on a plateau where we can go faster )
            for idx in range( len( greater_zero_idxs[ 0 ] ) ):
                r = greater_zero_idxs[ 0 ][ idx ]
                c = greater_zero_idxs[ 1 ][ idx ]
                step_size[ r, c ] = min( step_size[ r, c ] * self.rate_pos,
                                         self.max_step )

            # direction changed -> decrease step size
            # ( it seems we were too fast and jumped on the other side of 
            # the minimum valley )
            for idx in range( len( less_than_zero_idxs[ 0 ] ) ):
                r = less_than_zero_idxs[ 0 ][ idx ]
                c = less_than_zero_idxs[ 1 ][ idx ]
                step_size[ r, c ] = max( step_size[ r, c ] * self.rate_neg,
                                         self.min_step )

            change = -step_size * np.sign( gradient )
            weight += change
            last_gradient = np.copy( gradient )

        return weights


class IRpropPlus( AbstractUpdateMethod ):
    def __init__( self, layer_sizes, rate_pos=1.2, rate_neg=0.5,
                  init_step=0.0001, min_step=0.000001, max_step=50. ):
        self.rate_pos = rate_pos
        self.rate_neg = rate_neg
        self.max_step = max_step
        self.min_step = min_step
        self.last_gradient = [ ]
        self.last_error = -1
        self.step_size = [ ]
        self.last_change = [ ]
        # initialize last gradient and step size for each layer and weight
        for layer in range( len( layer_sizes ) - 1 ):
            init = np.zeros(
                (layer_sizes[ layer ] + 1, layer_sizes[ layer + 1 ]) )
            self.last_gradient.append( init )
            self.last_change.append( init )
            # set step size to given initial step size
            self.step_size.append( (init + 1) * init_step )

    def perform_update( self, weights, gradients, error ):
        if self.last_error == -1:
            self.last_error = error
        for i in range( len( weights ) ):
            last_gradient = self.last_gradient[ i ]
            last_change = self.last_change[ i ]
            step_size = self.step_size[ i ]
            gradient = gradients[ i ]
            weight = weights[ i ]

            # calculate the change in the gradient direction
            change = np.sign( gradient * last_gradient )

            # get the weights where:
            # -  change > 0 -> direction didn't change
            #    -  change < 0 -> direction changed
            #    - change == 0 -> one of the gradients is 0, do nothing
            greater_zero_idxs = np.where( change > 0 )
            less_than_zero_idxs = np.where( change < 0 )
            equals_zero_idxs = np.where( change == 0 )

            # direction didn't change -> increase step size
            # ( probably we are on a plateau where we can go faster )
            for idx in range( len( greater_zero_idxs[ 0 ] ) ):
                r = greater_zero_idxs[ 0 ][ idx ]
                c = greater_zero_idxs[ 1 ][ idx ]
                step_size[ r, c ] = min( step_size[ r, c ] * self.rate_pos,
                                         self.max_step )
                change = -step_size[ r, c ] * np.sign( gradient[ r, c ] )
                weight[ r, c ] += change
                last_change[ r, c ] = change
                last_gradient[ r, c ] = gradient[ r, c ]

            # direction changed -> decrease step size
            # ( it seems we were too fast and jumped on the other side of 
            # the minimum valley )
            for idx in range( len( less_than_zero_idxs[ 0 ] ) ):
                r = less_than_zero_idxs[ 0 ][ idx ]
                c = less_than_zero_idxs[ 1 ][ idx ]
                # decrease step size
                step_size[ r, c ] = max( step_size[ r, c ] * self.rate_neg,
                                         self.min_step )
                # if the error increased, revert last change
                if error > self.last_error:
                    weight[ r, c ] -= last_change[ r, c ]
                last_gradient[ r, c ] = 0

            # either the last or the current gradient is zero, that's not too
            # bad, so let's just go on and keep the step size
            for idx in range( len( equals_zero_idxs[ 0 ] ) ):
                r = equals_zero_idxs[ 0 ][ idx ]
                c = equals_zero_idxs[ 1 ][ idx ]
                change = -step_size[ r, c ] * np.sign( gradient[ r, c ] )
                weight[ r, c ] += change
                last_change[ r, c ] = change
                last_gradient[ r, c ] = gradient[ r, c ]

        self.last_error = error
        return weights


class PredictionNN( AbstractAlgorithm ):
    '''
    Predictive Feed Forward Neural Network Class
    '''

    def __init__( self, arrLayerSizes, iterations=1,
                  update_method=SimpleUpdate( 0.5 ), batch_update_size=1,
                  activation_function=nputils.sigmoidNPArray,
                  deriv_activation_function=nputils.sigmoidDeriv ):
        '''
        Creates a Prediction Neural Network - weights 
        @param arrLayerSizes: list with number of neurons each layer should 
        have. index starts with input layer.
        @param iterations: optional (default is 1), number of iterations 
        performed on the given data set when doing training
        @param update_method: optional (default is SimpleUpdate with lerning 
        rate of 0.5)Update method to be used for weight update
        @param batch_update_size: optional (default is 1), specifies the number
        of training examples to look at before applying a weight update. A size
        of 1 is usually referred to as stochastic (or incremental) gradient 
        descent whereas a value greater 1 is known as batch gradient descent.
        '''
        # Sizes for each layer, 0 is input layer
        self.arrLayerSizes = arrLayerSizes
        self.nrLayers = len( arrLayerSizes )
        self.iterations = iterations
        self.batch_update_size = batch_update_size
        self.update_method = update_method
        self.activation_function = activation_function
        self.deriv_activation_function = deriv_activation_function

        weightsArr = [ ]
        for layer in range( len( arrLayerSizes ) - 1 ):
            # weight matrix shape is first layer * second layer
            # bias term added on first dimension
            # generate random weights in the range of [-0.5, 0.5]
            weights = np.random.rand( arrLayerSizes[ layer ] + 1,
                                      arrLayerSizes[ layer + 1 ] ) - 0.5
            weightsArr.append( weights )

        self.weightsArr = weightsArr


    def train( self, dataSet ):
        '''
        Online Training for given dataset
        @param _dataSet: NumericalDataSet
        '''
        for it in range( self.iterations ):
            # randomly select observations as many times as there are
            # observations
            logging.info( "Iteration #" + str( it + 1 ) )
            it_error = 0
            batch_error = 0
            batch_updates = [ np.zeros( np.shape( w ) ) for w in
                              self.weightsArr ]
            for o in range( dataSet.get_nr_observations( ) ):
                inputArr, targetArr = dataSet.rand_observation( )
                # feed-forward
                outputs = self.feedforward( inputArr )
                current_error = self.calculate_error( targetArr, outputs[ -1 ] )
                it_error += current_error
                batch_error += current_error
                # backpropagation
                deltas = self.backpropagation( outputs, targetArr )
                weight_updates = self.calculate_weight_updates( deltas,
                                                                outputs )
                # accumulate weight updates until batch size is reached, then
                # do the weight update
                for i in range( len( self.weightsArr ) ):
                    batch_updates[ i ] += weight_updates[ i ]
                if (o + 1) % self.batch_update_size == 0:
                    self.update_method.perform_update( self.weightsArr,
                                                       batch_updates,
                                                       batch_error )
                    # print("  Avg. error: " + str(batch_error/self.batch_update_size) + "\n")
                    batch_error = 0
                    for j in range( len( batch_updates ) ):
                        batch_updates[ j ].fill( 0 )

            logging.info("  Avg. error: " + str(it_error / dataSet.get_nr_observations()) + "\n")


    def predict( self, dataSet ):
        '''
        Predicts targets for given data set
        @param _dataSet: data Set inheriting AbstractDataSet
        @return: outputs from the feed forward on each row 
        @rtype: list of numpy.ndarray (nr_obs * nr_output_neurons)
        '''
        predictions = [ ];
        # loop through dataset
        for observation, _ in dataSet.gen_observations( ):
            # make sure it numpy array
            inputArr = np.array( observation )
            # feedforward
            outputs = self.feedforward( inputArr )
            predictions.append( outputs[ -1 ] )

        return predictions


    def set_params( self, parameters ):
        '''Set parameters of predefined model(shape of parameters already specified)
        @param parameters: array of np.array
        @raise exception: if given parameters don't match in shape with model
        '''
        for wIndex in range( len( parameters ) ):
            if self.weightsArr[ wIndex ].shape != parameters[ wIndex ].shape:
                raise Exception(
                    "overwriting parameters have not the same shape as the model (weight Matrix) " + str(
                        wIndex ) + ".\n        model: " + str( self.weightsArr[
                                                                   wIndex ].shape ) + "\n  overwriting: " + str(
                        parameters[ wIndex ].shape ) )
            self.weightsArr[ wIndex ] = parameters[ wIndex ]


    def feedforward( self, input_vec ):
        '''
        feed inputs forward through net.
        @param: input_vec nparray of inputs. Size defined by input layer. Row vector shape = (1,x) hint: np.array([[1,2,3]])
        @return: activations for each layer shape = (1,x).
        @rtype: array of np.Arrays(1dim), for each layer one (weight layers + 1)
        @raise exception: if given input size doesn't match with input layer
        '''

        if (input_vec.shape != (1, self.arrLayerSizes[ 0 ])):
            raise Exception( "Invalid inputvector shape. (1," + str(
                self.arrLayerSizes[ 0 ] ) + ") expected, got " + str(
                input_vec.shape ) )

        outputs = [ input_vec ]

        # feed forward through network
        for i in range( len( self.weightsArr ) ):
            # input is output of last layer plus bias
            layer_in = nputils.addOneToVec( outputs[ -1 ] )
            # activation is weighted sum for each neuron
            layer_activation = np.dot( layer_in, self.weightsArr[ i ] )
            # activation function is a logistic unit, except last layer
            if i != len( self.weightsArr ) - 1:
                layer_out = self.activation_function( layer_activation )
            else:
                # max_idx = np.argmax(layer_activation)
                # layer_out = np.zeros(layer_activation.shape)
                # layer_out[0, max_idx] = 1.
                layer_out = layer_activation

            outputs.append( layer_out )

        return outputs


    def backpropagation( self, outputs, targets ):
        '''
        Propagates errors through NN, computing the partial gradients
        @param activations: List of np.arrays(1,x) obtained from feedforward
        @param targets: np.array of shape (1,output_layer_size) representing the desired output
        @return: list of deltas for each weight matrix
        '''
        # the deltas from the delta rule
        # deltas for the output layer - no sigmoid derivative since output is
        # linear deltas will have same shape as activations = (1,x)
        deltas = [ outputs[ -1 ] - targets ]
        # starting from second last layer, iterating backwards through nn UNTIL
        # second layer (input layer doesnt need deltas)
        # weights i are between layer i and i + 1
        for i in reversed( range( 1, self.nrLayers - 1 ) ):
            # multiply weights with previous computed deltas (first in list) to
            # obtain the sum over all neurons in next layer, for each neuron
            # in current layer
            sums = np.dot( self.weightsArr[ i ], deltas[ 0 ].transpose( ) )
            # remove last sum since it is from bias neuron. we don't need a
            # delta for it, since it doesn't have connections to the previous
            # layer
            sums = sums[ :-1, : ].transpose( )
            # element-wise multiply with the sigmoidal derivative for activation
            current_delta = self.deriv_activation_function(
                outputs[ i ] ) * sums
            # PREpend delta to array
            deltas.insert( 0, current_delta )
        return deltas


    def calculate_weight_updates( self, deltas, outputs ):
        '''
        Calculates updates for the weights based on the given deltas and activations.
        @param deltas: List of deltas calculated by backpropagation
        @param activations: List of activations calculated by feed forward  
        @return: List of weight updates
        '''
        changes = [ ]
        # weights i are between layer i and i + 1
        for i in range( self.nrLayers - 1 ):
            # here the activations need the additional bias neuron -> addOneToVec
            # unfortunately both arrays have to be transposed
            changes.append(np.dot( nputils.addOneToVec( outputs[ i ] ).transpose( ), deltas[ i ] ) )
        return changes

    def calculate_error( self, expected, actual ):
        return np.sum( 0.5 * np.power( expected - actual, 2 ) )


def convert_targets( targets ):
    result = np.zeros( (len( targets ), 10) )
    for i in range( len( targets ) ):
        result[ i, targets[ i ] ] = 1
    return np.array( result )


if __name__ == '__main__':

    training_data = np.loadtxt( '../../data/pendigits-training.txt' )[ :500, : ]
    testing_data = np.loadtxt( '../../data/pendigits-testing.txt' )

    layer_sizes = [ 16, 16, 10 ]
    update_method = Rprop( layer_sizes, init_step=0.01 )
    nn = PredictionNN( layer_sizes, iterations=300,
                       update_method=update_method,
                       batch_update_size=30,
                       activation_function=np.tanh,
                       deriv_activation_function=nputils.tanhDeriv )

    training_targets = convert_targets( training_data[ :, -1 ] )
    training_input = training_data[ :, 0:-1 ]
    maxs = np.max( training_input, axis=0 )
    mins = np.min( training_input, axis=0 )
    normalized_training_input = np.array(
        [ (r - mins) / (maxs - mins) for r in training_input ] )

    training_data_set = NumericalDataSet( normalized_training_input,
                                          training_targets )

    testing_targets = convert_targets( testing_data[ :, -1 ] )
    testing_input = testing_data[ :, 0:-1 ]
    maxs = np.max( testing_input, axis=0 )
    mins = np.min( testing_input, axis=0 )
    normalized_testing_input = np.array(
        [ (r - mins) / (maxs - mins) for r in testing_input ] )

    testing_data_set = NumericalDataSet( normalized_testing_input,
                                         testing_targets )

    nn.train( training_data_set )
    predictions = nn.predict( testing_data_set )

    predictions = [ np.argmax( p ) for p in predictions ]

    conf_matrix = np.zeros( (10, 10) )
    conf_matrix = np.concatenate( ( [ np.arange( 0, 10 ) ], conf_matrix ),
                                  axis=0 )
    conf_matrix = np.concatenate(
        ( np.transpose( [ np.arange( -1, 10 ) ] ), conf_matrix ), axis=1 )
    targets = testing_data[ :, -1 ]
    for i in range( len( targets ) ):
        conf_matrix[ targets[ i ] + 1, predictions[ i ] + 1 ] += 1

    print("Detection rate: " + str(
        np.sum( np.diagonal( conf_matrix[ 1:, 1: ] ) ) / len( targets ) ))
    print(str( conf_matrix ))
