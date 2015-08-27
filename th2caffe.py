#########################
## th2caffe #############
## Jarvis Du ############
#########################
## caffe loader from prototxt file ############
## layer pararmeters saver ####################
###############################################


def buildModel(prototxt_name, model_type, params_name, output_loc):
    # necessary imports for caffe
    import sys
    sys.path.append('/opt/caffe/python')
    import caffe
    import h5py
    # build model in caffe
    if model_type == 'test':
        net = caffe.Net(prototxt_name, caffe.TEST)
    else:
        net = caffe.Net(prototxt_name, caffe.TRAIN)
    # read parameters file
    params = h5py.File(params_name, 'r')
    # plug in parameters
    for key in net.params.keys():
        try:
            print('[Layer parameters] Now processing: layer '+ key)
            curWeights = params['weights/' + key].value
            curBias = params['bias/' + key].value
            net.params[key][0].data[:] = curWeights.reshape(net.params[key][0].data.shape)
            net.params[key][1].data[:] = curBias.reshape(net.params[key][1].data.shape)
        except KeyError:
            print('Layer "' + key + '": ' + 'No parameters needed.\n')
            sys.exit(0)
    # save caffemodel
    net.save(output_loc)
    print('--- Loading in python succeeded. .caffemodel file saved.')

import sys
buildModel(*sys.argv[1:])        
