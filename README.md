Attention-based Deep Multiple Instance Learning
================================================

by Maximilian Ilse (<ilse.maximilian@gmail.com>), Jakub M. Tomczak (<jakubmkt@gmail.com>) and Max Welling

Modifications (by nzw0301)
-------------------------
This repo is a fork of [Attention-based Deep Multiple Instance Learning](https://github.com/AMLab-Amsterdam/AttentionDeepMIL).
Main differences are

- Support `PyTorch 1.3.1` (I suppose this repo works by using newer PyTorch as well)
- Tiny refactorings

Examples (by nzw0301)
---------------------

The default model is based on an attention model without gated attention:

```bash
$ python main.py
Load Train and Test Set
Init Model
Start Training
Epoch: 1, Loss: 0.6876, Train error: 0.3850
Epoch: 2, Loss: 0.6480, Train error: 0.3700
Epoch: 3, Loss: 0.4586, Train error: 0.2250
Epoch: 4, Loss: 0.2609, Train error: 0.1050
Epoch: 5, Loss: 0.1443, Train error: 0.0450
Epoch: 6, Loss: 0.1565, Train error: 0.0500
Epoch: 7, Loss: 0.0774, Train error: 0.0350
Epoch: 8, Loss: 0.0489, Train error: 0.0150
Epoch: 9, Loss: 0.0938, Train error: 0.0150
Epoch: 10, Loss: 0.0196, Train error: 0.0050
Epoch: 11, Loss: 0.0698, Train error: 0.0250
Epoch: 12, Loss: 0.0402, Train error: 0.0100
Epoch: 13, Loss: 0.0091, Train error: 0.0000
Epoch: 14, Loss: 0.0052, Train error: 0.0000
Epoch: 15, Loss: 0.0182, Train error: 0.0100
Epoch: 16, Loss: 0.0010, Train error: 0.0000
Epoch: 17, Loss: 0.0005, Train error: 0.0000
Epoch: 18, Loss: 0.0003, Train error: 0.0000
Epoch: 19, Loss: 0.0002, Train error: 0.0000
Epoch: 20, Loss: 0.0002, Train error: 0.0000
Start Testing

True Bag Label, Predicted Bag Label: (1.0, 1)
True Instance Labels, Attention Weights: [(0.0, 0.0), (1.0, 0.513), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.486)]

True Bag Label, Predicted Bag Label: (1.0, 1)
True Instance Labels, Attention Weights: [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.497), (0.0, 0.0), (1.0, 0.503), (0.0, 0.0), (0.0, 0.0)]

True Bag Label, Predicted Bag Label: (1.0, 1)
True Instance Labels, Attention Weights: [(0.0, 0.007), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.002), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.991), (0.0, 0.0)]

True Bag Label, Predicted Bag Label: (1.0, 1)
True Instance Labels, Attention Weights: [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 1.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]

True Bag Label, Predicted Bag Label: (1.0, 1)
True Instance Labels, Attention Weights: [(0.0, 0.0), (1.0, 1.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]

Test Set, Loss: 0.5433, Test error: 0.1000
```

---

Pass `--gated` flag when you use the gated attention model:

```bash
$ python main.py --gated
Load Train and Test Set
Init Model
Start Training
Epoch: 1, Loss: 0.6758, Train error: 0.3850
Epoch: 2, Loss: 0.6116, Train error: 0.3250
Epoch: 3, Loss: 0.4499, Train error: 0.1800
Epoch: 4, Loss: 0.2565, Train error: 0.1100
Epoch: 5, Loss: 0.1892, Train error: 0.0650
Epoch: 6, Loss: 0.1374, Train error: 0.0450
Epoch: 7, Loss: 0.1161, Train error: 0.0350
Epoch: 8, Loss: 0.0611, Train error: 0.0300
Epoch: 9, Loss: 0.1216, Train error: 0.0450
Epoch: 10, Loss: 0.0685, Train error: 0.0200
Epoch: 11, Loss: 0.0515, Train error: 0.0250
Epoch: 12, Loss: 0.0219, Train error: 0.0100
Epoch: 13, Loss: 0.0019, Train error: 0.0000
Epoch: 14, Loss: 0.0006, Train error: 0.0000
Epoch: 15, Loss: 0.0004, Train error: 0.0000
Epoch: 16, Loss: 0.0003, Train error: 0.0000
Epoch: 17, Loss: 0.0002, Train error: 0.0000
Epoch: 18, Loss: 0.0002, Train error: 0.0000
Epoch: 19, Loss: 0.0002, Train error: 0.0000
Epoch: 20, Loss: 0.0001, Train error: 0.0000
Start Testing

True Bag Label, Predicted Bag Label: (1.0, 1)
True Instance Labels, Attention Weights: [(0.0, 0.0), (1.0, 0.627), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.005), (0.0, 0.0), (1.0, 0.368)]

True Bag Label, Predicted Bag Label: (1.0, 1)
True Instance Labels, Attention Weights: [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.481), (0.0, 0.0), (1.0, 0.519), (0.0, 0.0), (0.0, 0.0)]

True Bag Label, Predicted Bag Label: (1.0, 1)
True Instance Labels, Attention Weights: [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.999), (0.0, 0.0)]

True Bag Label, Predicted Bag Label: (1.0, 1)
True Instance Labels, Attention Weights: [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 1.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]

True Bag Label, Predicted Bag Label: (1.0, 1)
True Instance Labels, Attention Weights: [(0.0, 0.0), (1.0, 0.999), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]

Test Set, Loss: 0.2884, Test error: 0.0600
```

Overview
--------

PyTorch implementation of our paper "Attention-based Deep Multiple Instance Learning":
* Ilse, M., Tomczak, J. M., & Welling, M. (2018). Attention-based Deep Multiple Instance Learning. arXiv preprint arXiv:1802.04712. [link](https://arxiv.org/pdf/1802.04712.pdf).


Installation
------------

Installing PyTorch 1.3.1, using pip or conda, should resolve all dependencies.
Tested with Python 3.7.
Tested on both CPU and GPU.


Content
--------

The code can be used to run the MNIST-BAGS experiment, see Section 4.2 and Figure 1 in our [paper](https://arxiv.org/pdf/1802.04712.pdf).
In order to have a small and concise experimental setup, the code has the following limitation:
+ Mean bag length parameter shouldn't be much larger than 10, for larger numbers the training dataset will become unbalanced very quickly. You can run the data loader on its own to check, see __main__ part of `dataloader.py`
+ No validation set is used during training, no early stopping

__NOTE__: In order to run experiments on the histopathology datasets, please download datasets [Breast Cancer](http://bioimage.ucsb.edu/research/bio-segmentation) and [Colon Cancer](https://warwick.ac.uk/fac/sci/dcs/research/tia/data/crchistolabelednucleihe/). In the histopathology experiments we used a similar model to the model in `model.py`, please see the [paper](https://arxiv.org/pdf/1802.04712.pdf) for details.


How to Use
----------
`dataloader.py`: Generates training and test set by combining multiple MNIST images to bags. A bag is given a positive label if it contains one or more images with the label specified by the variable target_number.
If run as main, it computes the ratio of positive bags as well as the mean, max and min value for the number per instances in a bag.

`mnist_bags_loader.py`: Added the original data loader we used in the experiments. It can handle any bag length without the dataset becoming unbalanced. It is most probably not the most efficient way to create the bags. Furthermore it is only test for the case that the target number is ‘9’.

`main.py`: Trains a small CNN with the Adam optimization algorithm.
The training takes 20 epochs. Last, the accuracy and loss of the model on the test set is computed.
In addition, a subset of the bags labels and instance labels are printed.

`model.py`: The model is a modified LeNet-5, see <http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf>.
The Attention-based MIL pooling is located before the last layer of the model.
The objective function is the negative log-likelihood of the Bernoulli distribution.


Questions and Issues
--------------------

If you find any bugs or have any questions about this code please contact Maximilian or Jakub. We cannot guarantee any support for this software.

Citation
--------------------

Please cite our paper if you use this code in your research:
```
@article{ITW:2018,
  title={Attention-based Deep Multiple Instance Learning},
  author={Ilse, Maximilian and Tomczak, Jakub M and Welling, Max},
  journal={arXiv preprint arXiv:1802.04712},
  year={2018}
}
```

Acknowledgements
--------------------

The work conducted by Maximilian Ilse was funded by the Nederlandse Organisatie voor Wetenschappelijk Onderzoek (Grant DLMedIa: Deep Learning for Medical Image Analysis).

The work conducted by Jakub Tomczak was funded by the European Commission within the Marie Skodowska-Curie Individual Fellowship (Grant No. 702666, ”Deep learning and Bayesian inference for medical imaging”).
