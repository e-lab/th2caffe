#th2caffe

A torch-nn to caffe converter for specific layers.

## Contents
 - [Installation](#installation)
 - [Supported layers](#supported-layers)
 - [How to run](#how-to-run)
 - [Output](#output)

## Installation
`th2caffe` requires the following packages.

`caffe`: To solve the discrepancy between the size of `nn.SpatialMaxPooling` and `caffe:Pooling(MAX)`, some `caffe` codes are modified. This special caffe version can be installed either by overwriting some source codes in the original `caffe`, or by installing a [folk version](https://github.com/dujianyi/caffe) of caffe.

For overwriting files, please run
```bash
cp caffe/caffe.proto location/to/caffe/src/caffe/proto/caffe.proto
cp caffe/pooling_layer.cpp location/to/caffe/src/caffe/layers/pooling_layer.cpp
cp caffe/vision_layers.hpp location/to/caffe/include/caffe/vision_layers.hpp
make clean
make all
make test
make runtest
```

See `BVLC/caffe`'s [GitHub](https://github.com/BVLC/caffe) for installation. Install `pycaffe` also to run python scripts.

Please note:

`hdf5`: Install both versions of python and torch.
for python, run
```bash
pip install h5py
```
for torch, install [deepmind](https://github.com/deepmind/torch-hdf5)'s version following the instructions.

## Supported layers
Currently `th2caffe` supports the following layers for conversions.
```lua
nn.Linear                          (caffe: InnerProduct)
nn.SpatialConvolutionMM            (caffe: Convolution)
nn.SpatialMaxPooling (floor/ceil)  (caffe: Pooling, type MAX)
nn.ReLU                            (caffe: ReLU)
nn.View                            (caffe: Reshape)
nn.Dropout                         (caffe: Dropout)
nn.SoftMax                         (caffe: Softmax)
```
## How to run
```bash
cd th2caffe
th th2caffe.lua --nf netFile \
                --name nnName \
                --c numChannels \
                --w width \
                --h height \
                --loc location \
                --caffe caffeloc
```

with arguments:

```
--nf netFile       Path to the .net file saving all information for a torch neural network
--name nnName      Name that is specified to the caffe neural network
--c numChannels    Number of input channels
--w width          Width of input image
--h height         Height of input image
--loc location     Location to save the output files (will be created if it doesn't exist)
--caffe caffeloc   Location to the caffe source
```
## Output
The output files include:
```
location/architecture/deploy.prototxt   Prototxt file saving the architecture
location/params/params.h5               .h5 files saving layer parameters
location/params/params.caffemodel       .caffemodel file that can be loaded by caffe
```

### License

MIT
