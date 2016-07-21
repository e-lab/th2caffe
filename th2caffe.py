#########################
## th2caffe #############
## Jarvis Du ############
###############################################
## caffe loader from prototxt file ############
## layer pararmeters saver ####################
###############################################


def buildModel(prototxt_name, model_type, params_name, output_loc, caffe_loc):
    # necessary imports for caffe
    import sys
    sys.path.append(caffe_loc+'/python')
    try: 
        import caffe
    except ImportError:
        print('pyCAFFE: Not found!') 
        sys.exit(0)
    import h5py
    # build model in caffe
    if model_type == 'test':
        net = caffe.Net(prototxt_name, caffe.TEST)
    else:
        net = caffe.Net(prototxt_name, caffe.TRAIN)
    # read parameters file
    params = h5py.File(params_name, 'r')
    # plug in parameters
    for key in net.params:
        print('[Layer parameters] Now processing: layer '+ key)
        try:
            weights = params['weights/' + key].value
            print('    weights')
            net.params[key][0].data[:] = weights.reshape(net.params[key][0].data.shape)
        except KeyError:
            pass
        try:
            bias = params['bias/' + key].value
            print('    bias')
            net.params[key][1].data[:] = bias.reshape(net.params[key][1].data.shape)
        except KeyError:
            pass
        try:
            mean = params['mean/' + key].value
            print('    mean')
            net.params[key][0].data[:] = mean.reshape(net.params[key][0].data.shape)
        except KeyError:
            pass
        try:
            var = params['var/' + key].value
            print('    var')
            net.params[key][1].data[:] = var.reshape(net.params[key][1].data.shape)
            net.params[key][2].data[0] = 1
        except KeyError:
            pass
    # save caffemodel
    net.save(output_loc)
    print('--- Loading in python succeeded. .caffemodel file saved.')

import sys
buildModel(*sys.argv[1:])        
