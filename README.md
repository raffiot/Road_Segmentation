Deep neural network implementation for road recognition.  
For more details about implementation and results please look at road-segmentation-convolutional.pdf .

Libraries:  
 The following libraries have to be installed to be able to run the project:  
  Keras 2.1.2  
  CudNN 9.0  
  Cuda 6.0  
  Tensorflow 1.4.0  
  hpy5 2.7.1  
  
We create our neural net thanks to Keras and hpy5 permit to save the weights  
and to load them back in the neural network.  

Setup:  
  
How to run:  
In order to run the code faster we already provide you the weights that made our result  
(the file 'weights-best-submission.hdf5') then directory hierarchy in order to run 'run.py' have to be  
Project  
 |  
 |--run.py  
 |--weights-best-submission.hdf5  
 |--helpers.py  
 |--image_augmentation.py  
 |--prediction_with_weights.py  
 |--training.py  
 |  
 +--test_set_images  
 | |  
 | +--test_1  
 | | |  
 | | |--test_1.png  
 | +--test_2  
 | | |  
 | | |--test_2.png  
 ...  
 | +--test_50  
 | | |  
 | | |--test_50.png  
 +--training  
 | |  
 | +--groundtruth  
 | | |  
 | | |--satImage_001.png  
 | | |--satImage_002.png  
 ...  
 | | |--satImage_100.png  
 | +--images  
 | | |  
 | | |--satImage_001.png  
 | | |--satImage_002.png  
 ...  
 | | |--satImage_100.png  
  
A new directory predictions should have been created.  
