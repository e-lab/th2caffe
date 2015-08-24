#th2caffe

A torch-nn to caffe converter for specific layers.

## Contents
 - [Installation](#installation)
 - [Supported layers](#supported-layers)
 - [How to run](#how-to-run)
 - [Output](#output)

## Installation
`th2caffe` requires the following packages.

`caffe`: See their [GitHub](https://github.com/BVLC/caffe) for installation, install `pycaffe` also to run python scripts

Please note:
- Specify the installing location to `/opt/caffe`, or change to the corresponding folder in th2caffe.py (see comments). Or create link in `/opt` as
```bash
sudo ln -s /path/to/caffe /opt/caffe
```

- Override `pooling_layer.cpp` in `caffe/src/caffe/layers` with the file given and recompile caffe.
```bash
cd /opt/caffe

make clean

make all

make test

make runtest
```

The reason to recompile caffe is that caffe takes `ceil` during calculation of `MaxPooling` size, which leads to discrepancy with torch. Recompilation of caffe can lead to failure in several runtests, but from what is tested, there is no function affected.

`hdf5`: Install both versions of python and torch.
for python, run
```bash
pip install h5py
```
for torch, install [deepmind](https://github.com/deepmind/torch-hdf5)'s version following the instructions.

## Supported layers
Currently `th2caffe` supports the following layers for conversions.

- `nn.Linear`                 (caffe: `InnerProduct`)
- `nn.SpatialConvolutionMM`   (caffe: `Convolution`)
- `nn.SpatialMaxPooling`      (caffe: `Pooling`, type `MAX`)
- `nn.ReLU`                   (caffe: `ReLU`)
- `nn.View`                   (caffe: `Reshape`)
- `nn.Dropout`                (caffe: `Dropout`)
- `nn.SoftMax`                (caffe: `Softmax`)

## How to run
```bash
cd th2caffe
th th2caffe.lua --nf netFile --name nnName --c numChannels --w width, --h height, --loc location)
```

with arguments:

```
--nf netFile       Path to the .net file saving all information for a torch neural network

--name nnName      Name that is specified to the caffe neural network

--c numChannels    Number of input channels

--w width          Width of input image

--h height         Height of input image

--loc location     Location to save the output files (will be created if it doesn't exist)
```
## Output
The output files include:
```
location/architecture/deploy.prototxt   Prototxt file saving the architecture

location/params/params.h5               .h5 files saving layer parameters

location/params/params.caffemodel       .caffemodel file that can be loaded by caffe
```
