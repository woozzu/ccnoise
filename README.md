# Cross-Channel Image Noise Model
For more information, please visit our [project page](http://snam.ml/research/ccnoise).

To run demo,
- Create a dataset from temporal images: run `demo_dataset.m`
- Create training data: run `train/demo_create_training_data.py`
- Train a multi-layer perceptron (MLP): run `caffe.exe train -solver=train/solver.prototxt`
- Estimate the noise parameters of a test image using a trained MLP: run `demo_estimation.m`

Note that the example data is only for demo and may not be enough to reproduce our work. To do this, you should take many temporal images (for example, 500, 1000, ...) or download [our dataset](http://snam.ml/research/ccnoise).

# Citation
Please cite the following paper in your publications if you use our cross-channel image noise model:

    @inproceedings{nam2016holistic,
      title={A Holistic Approach to Cross-Channel Image Noise Modeling and Its Application to Image Denoising},
      author={Nam, Seonghyeon and Hwang, Youngbae and Matsushita, Yasuyuki and Kim, Seon Joo},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
      pages={1683--1691},
      year={2016}
    }