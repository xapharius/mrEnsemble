from engine.constants.run_type import HADOOP, LOCAL, INLINE
from engine.engine import Engine
from datahandler.image.image_data_handler import ImageDataHandler
from algorithms.image.gaussian_filter_factory import GaussianFilterFactory
from skimage import io as skio


if __name__ == '__main__':
    
    
    print("=== Gaussian Filter Example ===")
    
    sigma = 2.
    run_type = HADOOP
    data_file = 'hdfs:///user/linda/ml/data/test-images' if run_type == HADOOP else '../data/test-images'
    
    print(  "\n      data: " + data_file
          + "\n     sigma: " + str(sigma)
          + "\n  run type: " + run_type
          + "\n"
          )
    
    # 1. define algorithm
    pred_nn = GaussianFilterFactory(sigma)
    
    # 2. set data handler (pre-processing, normalization, data set creation)
    data_handler = ImageDataHandler()
    
    # 3. run
    engine = Engine(pred_nn, data_file, data_handler=data_handler, verbose=True)
    gaussian_filter = engine.start(_run_type=run_type)
    
    skio.imsave('cat_gaussian.png', gaussian_filter.filtered[0])
