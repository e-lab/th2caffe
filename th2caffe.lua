----------------------------
-- th2caffe ----------------
-- Jarvis Du ---------------
-- Aug 20, 2015 ------------
----------------------------
-- Updated:
-- Jul 8, 2016 Marko Vitez -
----------------------------
-- A converter from torch to caffe --------------
-- Loadcaffe helps a lot with the codes ---------
-------------------------------------------------

local curLayerNum
torch.setdefaulttensortype('torch.FloatTensor')

function layertype(mtype)

   if mtype == 'SpatialConvolution' or mtype == 'SpatialConvolutionMM' then
      return 'Convolution', 'convolution'
   elseif mtype == 'SpatialMaxPooling' or mtype == 'SpatialAveragePooling' then
      return 'Pooling', 'pooling'
   elseif mtype == 'SpatialBatchNormalization' then
      return 'BatchNorm', 'batch_norm'
   elseif mtype == 'ReLU' then
      return 'ReLU'
   elseif mtype == 'PReLU' then
      return 'PReLU'
   elseif mtype == 'View' then
      return 'Reshape', 'reshape'
   elseif mtype == 'Linear' then
      return 'InnerProduct', 'inner_product'
   elseif mtype == 'Dropout' then
      return 'Dropout'
   elseif mtype == 'SoftMax' then
      return 'Softmax', 'softmax'
   elseif mtype == 'JoinTable' then
      return 'Concat'
   elseif mtype == 'CAddTable' then
      return 'Bias', 'bias'
   elseif mtype == 'Identity' or mtype == 'Padding' then
      return nil
   else
      error('Unsupported layer: ' .. mtype .. '\n')
   end

end

function deployParameters(m, name)

   local mtype = torch.type(m):sub(4)
   if m.padH then
      deployFile:write('    pad_h: ' .. m.padH .. '\n')
      deployFile:write('    pad_w: ' .. m.padW .. '\n')
   end
   if m.kH then
      deployFile:write('    kernel_h: ' .. m.kH .. '\n')
      deployFile:write('    kernel_w: ' .. m.kW .. '\n')
   end
   if m.dH then
      deployFile:write('    stride_h: ' .. m.dH .. '\n')
      deployFile:write('    stride_w: ' .. m.dW .. '\n')
   end
   if mtype == 'SpatialMaxPooling' then
      deployFile:write('    pool: MAX\n')
   elseif mtype == 'SpatialAveragePooling' then
      deployFile:write('    pool: AVE\n')
   elseif mtype == 'SpatialConvolution' or mtype == 'SpatialConvolutionMM' then
      deployFile:write('    num_output: ' .. m.nOutputPlane .. '\n')
      if not m.bias then
         deployFile:write('    bias_term: false\n')
      end
   elseif mtype == 'SpatialBatchNormalization' then
      deployFile:write('    use_global_stats: 1\n')
      deployFile:write('    moving_average_fraction: ' .. (1-m.momentum) .. '\n')
      deployFile:write('    eps: ' .. m.eps .. '\n')
   elseif mtype == 'Linear' then
      innerproductSize = #m:parameters()[1]
      deployFile:write('    num_output: ' .. tostring(innerproductSize[1]) .. '\n')
      deployFile:write('    axis: 0\n')
   elseif mtype == 'SoftMax' then
      deployFile:write('    axis: 0\n')
   elseif mtype == 'View' then
      deployFile:write('    shape {\n')
      for isize = 1, #m.size do
         deployFile:write('      dim: ' .. m.size[isize] .. '\n')
      end
      deployFile:write('    }\n')
   elseif mtype == 'CAddTable' then
      deployFile:write('    axis: 0\n')
   end
   
end

function deployData(m, name)

   if m.running_mean then
      paramsFile:write('mean/' .. name, m.running_mean)
      paramsFile:write('var/' .. name, m.running_var)
   elseif m.weight then
      paramsFile:write('weights/' .. name, m.weight)
      if m.bias then
         paramsFile:write('bias/' .. name, m.bias)
      else
         m.bias = torch.zeros(m.weight:size(1))
         print('Adding bias to ' .. name)
      end
   end

end

function deployCAddTable(m, depends, shape)

   local tmpshape = shape[1]:clone()
   tmpshape[2] = shape[1][2] - shape[2][2]
   deployFile:write('layer {\n')
   deployFile:write('  type: "Slice"\n')
   deployFile:write('  name: "CAddTable/Slice' .. curLayerNum .. '"\n')
   deployFile:write('  top: "CAddTable/FirstSlice' .. curLayerNum .. '"\n')
   deployFile:write('  # outdim: ' .. vec2string(shape[2]) .. '\n')
   deployFile:write('  top: "CAddTable/SecondSlice' .. curLayerNum .. '"\n')
   deployFile:write('  # outdim: ' .. vec2string(tmpshape) .. '\n')
   deployFile:write('  bottom: "' .. depends[1] .. '"\n')
   deployFile:write('  slice_param {\n')
   deployFile:write('    slice_point: ' .. shape[2][2] .. '\n')
   deployFile:write('  }\n')    
   deployFile:write('}\n')
   
   deployFile:write('layer {\n')
   deployFile:write('  type: "Bias"\n')
   deployFile:write('  name: "CAddTable/Bias' .. curLayerNum .. '"\n')
   deployFile:write('  top: "CAddTable/Bias' .. curLayerNum .. '"\n')
   deployFile:write('  # outdim: ' .. vec2string(shape[2]) .. '\n')
   deployFile:write('  bottom: "CAddTable/FirstSlice' .. curLayerNum .. '"\n')
   deployFile:write('  bottom: "' .. depends[2] .. '"\n')
   deployFile:write('  bias_param {\n')
   deployFile:write('    axis: 0\n')
   deployFile:write('  }\n')    
   deployFile:write('}\n')

   deployFile:write('layer {\n')
   deployFile:write('  type: "Concat"\n')
   deployFile:write('  name: "CAddTable/Concat' .. curLayerNum .. '"\n')
   deployFile:write('  top: "CAddTable/Concat' .. curLayerNum .. '"\n')
   deployFile:write('  # outdim: ' .. vec2string(shape[1]) .. '\n')
   print('CAddTable' .. curLayerNum .. ' --> (' .. vec2string(shape[1]) .. ')')
   deployFile:write('  bottom: "CAddTable/Bias' .. curLayerNum .. '"\n')
   deployFile:write('  bottom: "CAddTable/SecondSlice' .. curLayerNum .. '"\n')
   deployFile:write('}\n')
   
   return 'CAddTable/Concat' .. curLayerNum, shape[1]

end

function deployAffine(m, depends, shape)

   local name = 'SpatialBatchNormalization/Affine'.. curLayerNum

   deployFile:write('layer {\n')
   deployFile:write('  type: "Scale"\n')
   deployFile:write('  name: "' .. name .. '"\n')
   deployFile:write('  top: "' .. name .. '"\n')
   deployFile:write('  # outdim: ' .. vec2string(shape) .. '\n')
   print('SpatialBatchNormalization/Affine' .. curLayerNum .. ' --> (' .. vec2string(shape) .. ')')
   deployFile:write('  bottom: "SpatialBatchNormalization' .. curLayerNum .. '"\n')
   if m.bias then
      deployFile:write('  scale_param {\n')
      deployFile:write('    bias_term: true\n')
      deployFile:write('  }\n')
   end
   deployFile:write('}\n')
   
   paramsFile:write('weights/' .. name, m.weight)
   if m.bias then
      paramsFile:write('bias/' .. name, m.bias)
   end

   return name, shape

end

function adjustshape(m, shape)

   local mtype = torch.type(m):sub(4)
   if mtype == 'JoinTable' then
      newshape = shape[1]:clone()
      for i=2,#shape do
         newshape[m.dimension] = newshape[m.dimension] + shape[i][m.dimension]
      end
      return newshape
   elseif mtype == 'CAddTable' then
      for i=2,#shape do
         if (shape[i]-shape[1]):abs():max() > 0 then
            print('Shapes mismatch in CAddTable: ' .. vec2string(shape[1]) .. ' / ' .. vec2string(shape[i]))
         end
      end
      return shape[1]:clone()
   end
   shape = shape:clone()
   if mtype == 'SpatialConvolution' or mtype == 'SpatialConvolutionMM' then
      shape[2] = m.nOutputPlane
      shape[3] = torch.floor((shape[3] + 2*m.padH - m.kH) / m.dH + 1)
      shape[4] = torch.floor((shape[4] + 2*m.padW - m.kW) / m.dW + 1)
   elseif mtype == 'SpatialMaxPooling' or mtype == 'SpatialAveragePooling' then
      if m.ceil_mode then
         shape[3] = torch.ceil((shape[3] + 2*m.padW - m.kH)/m.dH + 1)
         shape[4] = torch.ceil((shape[4] + 2*m.padW - m.kW)/m.dW + 1)
      else
         shape[3] = torch.floor((shape[3] + 2*m.padW - m.kH)/m.dH + 1)
         shape[4] = torch.floor((shape[4] + 2*m.padW - m.kW)/m.dW + 1)
      end
   elseif mtype == 'View' then
      shape = torch.IntTensor({1, 1, 1, 1})
      for isize = 1, #m.size do
         shape[4-#m.size+isize] = m.size[isize]
      end
   end
   return shape

end

function vec2string(v)

   s = tostring(v[1])
   for i=2,v:size(1) do
      s = s .. ', ' .. v[i]
   end
   return s

end

function exportmodule(m, depends, shape)
  
   local mtype = torch.type(m):sub(4)
   if mtype == 'ConcatTable' then      
      local out = {}
      local newshape = {}
      for i=1,#m.modules do
         out[i], newshape[i] = exportmodule(m.modules[i], depends, shape)
      end
      return out, newshape
   elseif mtype == 'Sequential' then
      local out, shape = exportmodule(m.modules[1], depends, shape)
      for i=2,#m.modules do
         out, shape = exportmodule(m.modules[i], out, shape)
      end
      return out, shape
   else
      type_name, param_name = layertype(mtype)
      if not type_name then
         return depends, shape
      end
      curLayerNum = curLayerNum + 1
      -- Manage the case, where inputs to CAddTable are of different size
      -- We don't have padding, so slice it, sum and reconcatenate
      if mtype == 'CAddTable' and #shape == 2 and
         shape[1][1] == shape[2][1] and shape[1][2] ~= shape[2][2] and
         shape[1][3] == shape[2][3] and shape[1][4] == shape[2][4] then
         return deployCAddTable(m, depends, shape)
      end
      local name = mtype .. curLayerNum
      shape = adjustshape(m, shape, depends)
      deployFile:write('layer {\n')
      deployFile:write('  type: "' .. type_name .. '"\n')
      deployFile:write('  name: "' .. name .. '"\n')
      deployFile:write('  top: "' .. name .. '"\n')
      deployFile:write('  # outdim: ' .. vec2string(shape) .. '\n')
      print(name .. ' --> (' .. vec2string(shape) .. ')')
      if torch.type(depends) == 'string' then
         deployFile:write('  bottom: "' .. depends .. '"\n')
      else
         for i=1,#depends do
            deployFile:write('  bottom: "' .. depends[i] .. '"\n')
         end
      end
      if param_name then
         deployFile:write('  ' .. param_name .. '_param {\n')
         deployParameters(m, name)
         deployFile:write('  }\n')
      end
      deployFile:write('}\n')
      deployData(m, name)
      -- SpatialBatchNormalization needs other layers after itself,
      -- so deploy them
      if mtype == 'SpatialBatchNormalization' and m.affine then
         m.train = false
         return deployAffine(m, depends, shape)
      end
      return name, shape
   end

end

function Convert(netFile, nnName, inC, inW, inH, loc, caffeLoc)

   inputDim = torch.IntTensor({1, inC, inH, inW})
   nnToFile(netFile, nnName, inputDim, 'data', loc)
   caffeInPy(loc, caffeLoc)

end

-- Convert from nn to prototxt
function nnToFile(netFile, nnName, inputDim, nnInput, loc)

   print('--- Generate prototxt file into: ')
   -- Require necessary packages
   require 'nn'
   require 'hdf5'
   -- Read network from file
   nnSeq = torch.load(netFile)
   -- create new folders
   os.execute('mkdir ' .. loc)
   os.execute('mkdir ' .. loc .. '/architecture')
   os.execute('mkdir ' .. loc .. '/params')
   -- Prototxt global statement
   print(loc .. '/architecture/deploy.prototxt')
   deployFile = io.open(loc .. '/architecture/deploy.prototxt', 'w')
   deployFile:write('name: "' .. nnName .. '"\n')
   deployFile:write('input: "' .. nnInput .. '"\n')
   deployFile:write('input_shape {\n')
   for i = 1, 4 do
      deployFile:write('    dim: ' .. inputDim[i] .. '\n')
   end
   deployFile:write('}\n')
   -- hdf5 file statement
   paramsFile = hdf5.open(loc .. '/params/params.h5', 'w')
   -- Counter of layers
   curLayerNum = 0
   exportmodule(nnSeq, 'data', inputDim)
   deployFile:close()
   paramsFile:close()
   -- Uncomment if you want to save a Torch model with these modifications:
   -- bias added (otherwise nn does not work)
   -- SpatialBatchNormalization.train set to false in order to work properly
   --torch.save('updated-model.net', nnSeq)
   print('--- Prototxt file generated.')

end


-- Build Caffe model in Python and output binary weight file
function caffeInPy(loc, caffeLoc)

   prototxt_name = loc .. '/architecture/deploy.prototxt'
   params_name = loc .. '/params/params.h5'
   output_name = loc .. '/params/params.caffemodel'
   print('-- Load in python and execute ...')
   os.execute('python th2caffe.py "' .. prototxt_name .. '" "test" "' .. params_name .. '" "' .. output_name .. '" "' .. caffeLoc .. '"')

end

-- Main program
-- input arguments
title = 'th2caffe\n'
opt = lapp(title .. [[
--nf         (default none)         Path to .net file
--name       (default none)         Name of network in caffe
--c          (default 3)            Number of channels in input
--w          (default 224)          Width of input image
--h          (default 224)          Height of input image
--loc        (default ./test)       Location to save the outputs
--caffe      (default /opt/caffe)   Location of caffe source
]])

Convert(opt.nf, opt.name, opt.c, opt.w, opt.h, opt.loc, opt.caffe)