# Modularized SSD implementation in Tensorflow
This repo is an adaptation of [balancap/SSD-Tensorflow](https://github.com/balancap/SSD-Tensorflow/).
The original structure is maintained. All original training and evaluation parameters are unchanged.
For detail information and instructions please refer to the [original README file](./README_original.md).

The modular SSD model is implemented by:
* adding nets/modular_ssd.py - 
This file uses the same structure as ssd_vgg_300.py. ssd_net() is updated to enable interchanging.
* moving concrete definition of ssd parameters and network to nets/ssd_blocks.py.
* adding new flags 'feature_extractor' and 'model' to train_ssd_network.
* defining backbone network.

##HOWTO:
###Train and evaluate
See [original README file](./README_original.md) for help.
Commands specific to modular_ssd can be found in Modular_SSD_Commands.md
###Construct new network
Example of backbone network: [vgg.py](./nets/vgg.py).

The SSD blocks can be changed too. Just be careful with parameters.