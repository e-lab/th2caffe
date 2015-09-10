----------------------------
-- th2caffe ----------------
-- Jarvis Du ---------------
-- Aug 20, 2015 ------------
----------------------------
-- A converter from torch to caffe --------------
-- Loadcaffe helps a lot with the codes ---------
-------------------------------------------------

Convert = function(netFile, nnName, inC, inW, inH, loc, caffeLoc)
   inputDim = {1, inC, inH, inW}
   nnToFile(netFile, nnName, inputDim, 'data', loc)
   caffeInPy(loc, caffeLoc)
end

-- Convert from nn to prototxt
nnToFile = function(netFile, nnName, inputDim, nnInput, loc)
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
   for i = 1, 4 do
      deployFile:write('input_dim: ' .. inputDim[i] .. '\n')
   end
   --[[
   deployFile:write('input_dim: ' .. nnInput .. '\n')
   deployFile:write('input_dim: ' .. nnInput .. '\n')
   deployFile:write('input_dim: ' .. nnInput .. '\n')
   deployFile:write('input_dim: ' .. nnInput .. '\n')
   ]]
   -- hdf5 file statement
   paramsFile = hdf5.open(loc .. '/params/params.h5', 'w')
   -- Counter of layers
   curLayerNum = 0
   lastLayer = ''
   -- Generate names
   print('--- Inspecting every layer')
   names = {}
   bottom_names = {}
   top_names = {}
   for i = 1, #nnSeq do
      nnCur = nnSeq.modules[i]
      if (torch.type(nnCur) == 'nn.SpatialConvolutionMM') then
         print('[Layer ' .. tostring(i) .. '] Convolution')
         curLayerNum = curLayerNum + 1
         names[i] = 'conv' .. tostring(curLayerNum)
         if (curLayerNum == 1) then
            bottom_names[i] = nnInput
         else
            bottom_names[i] = lastLayer
         end
         top_names[i] = names[i]
         lastLayer = names[i]
         -- update correctDims
         if (not (nnCur.padding == nil)) then
            padH = nnCur.padding
            padW = nnCur.padding
         elseif (not (nnCur.padH == nil)) then
            padH = nnCur.padH
            padW = nnCur.padW
         else
            padH = 0
            padW = 0
         end
      end
      if (torch.type(nnCur) == 'nn.SpatialMaxPooling') then
         print('[Layer ' .. tostring(i) .. '] MaxPooling')
         names[i] = 'pool' .. tostring(curLayerNum)
         bottom_names[i] = lastLayer
         top_names[i] = names[i]
         lastLayer = names[i]
          -- update correctDims
         if (not (nnCur.padding == nil)) then
            padH = nnCur.padding
            padW = nnCur.padding
         elseif (not (nnCur.padH == nil)) then
            padH = nnCur.padH
            padW = nnCur.padW
         else
            padH = 0
            padW = 0
         end
      end
      if (torch.type(nnCur) == 'nn.ReLU') then 
         print('[Layer ' .. tostring(i) .. '] ReLU')
         names[i] = 'relu' .. tostring(curLayerNum)
         bottom_names[i] = lastLayer
         top_names[i] = bottom_names[i]
      end
      if (torch.type(nnCur) == 'nn.View') then
         print('[Layer ' .. tostring(i) .. '] Reshape')
         names[i] = 'reshape' .. tostring(curLayerNum)
         bottom_names[i] = lastLayer
         top_names[i] = names[i]
         lastLayer = names[i]
      end
      --[[
      if (torch.type(nnCur) == 'nn.Tanh') then 
         layerCount['tanh'] = layerCount['tanh'] + 1
         names[i] = 'tanh' .. tostring(layerCount['tanh'])
      end
      ]]
      --[[
      if (torch.type(nnCur) == 'nn.Sigmoid') then
         layerCount['sigm'] = layerCount['sigm'] + 1
         names[i] = 'sigm' .. tostring(layerCount['sigm'])
      end
      ]]
      if (torch.type(nnCur) == 'nn.Linear') then
         print('[Layer ' .. tostring(i) .. '] Linear')
         curLayerNum = curLayerNum + 1
         names[i] = 'ip' .. tostring(curLayerNum)
         if (curLayerNum == 1) then
            bottom_names[i] = nnInput
         else
            bottom_names[i] = lastLayer
         end
         top_names[i] = names[i]
         lastLayer = names[i]
      end
      --[[
      -- LRN
      ]]
      if (torch.type(nnCur) == 'nn.Dropout') then
         print('[Layer ' .. tostring(i) .. '] Dropout')
         names[i] = 'drop' .. tostring(curLayerNum)
         bottom_names[i] = lastLayer
         top_names[i] = bottom_names[i]
      end
      if (torch.type(nnCur) == 'nn.SoftMax') then
         print('[Layer ' .. tostring(i) .. '] SoftMax')
         names[i] = 'soft' .. tostring(curLayerNum)
         bottom_names[i] = lastLayer
         top_names[i] = names[i]
      end
   end
   -- Process every layer
   print('--- Now processing layers')
   -- Current input size
   curInput = inputDim
   ceilMode = 0
   for i = 1, #nnSeq do
      nnCur = nnSeq.modules[i]
      -- layer {
      deployFile:write('layer {\n')  -- Prototxt layer header
      -- Convolution layer
      if (torch.type(nnCur) == 'nn.SpatialConvolutionMM') then
         deployFile:write('  name: "' .. names[i] .. '"\n')
         deployFile:write('  type: "Convolution"\n')
         -- TOP/BOTTOM
         deployFile:write('  bottom: "' .. bottom_names[i] .. '"\n')
         deployFile:write('  top: "' .. top_names[i] .. '"\n')
         --[[
         -- blobs {
         deployFile:write('  blobs {\n')
         deployFile:write('    num: ' .. tostring(nnCur.nOutputPlane) .. '\n')
         deployFile:write('    channels: ' .. tostring(nnCur.nInputPlane) .. '\n')
         deployFile:write('  }\n')
         -- blobs }
         --]]
         -- convolution_param {
         deployFile:write('  convolution_param {\n')
         -- Omitted?
         deployFile:write('    num_output: ' .. tostring(nnCur.nOutputPlane) .. '\n')
         -- Omitted?
         ----[[
         if (not (nnCur.padding == nil)) then
            padH = nnCur.padding
            padW = nnCur.padding
         elseif (not (nnCur.padH == nil)) then
            padH = nnCur.padH
            padW = nnCur.padW
         else
            padH = 0
            padW = 0
         end
         deployFile:write('    pad_h: ' .. tostring(padH) .. '\n')
         deployFile:write('    pad_w: ' .. tostring(padW) .. '\n')
         ----]]
         --  deployFile:write('    pad: ' .. tostring(nnCur.padding) .. '\n')
         deployFile:write('    kernel_h: ' .. tostring(nnCur.kH) .. '\n')
         deployFile:write('    kernel_w: ' .. tostring(nnCur.kW) .. '\n')
         deployFile:write('    stride_h: ' .. tostring(nnCur.dH) .. '\n')
         deployFile:write('    stride_w: ' .. tostring(nnCur.dW) .. '\n')
         deployFile:write('  }\n')
         -- convolution_param }
         -- weights & bias
         paramsFile:write('weights/' .. names[i], nnCur:parameters()[1])
         paramsFile:write('bias/' .. names[i], nnCur:parameters()[2])
         -- Update output size
         tempInput = curInput
         curInput[2] = nnCur.nOutputPlane
         curInput[3] = torch.floor((tempInput[3] + 2*nnCur.padding - nnCur.kH)/nnCur.dH + 1) -- height
         curInput[4] = torch.floor((tempInput[4] + 2*nnCur.padding - nnCur.kW)/nnCur.dW + 1) -- width
      end
      -- Pooling
      if (torch.type(nnCur) == 'nn.SpatialMaxPooling') then
         if (not (nnCur.padding == nil)) then
            padH = nnCur.padding
            padW = nnCur.padding
         elseif (not (nnCur.padH == nil)) then
            padH = nnCur.padH
            padW = nnCur.padW
         else
            padH = 0
            padW = 0
         end 
         deployFile:write('  name: "' .. names[i] .. '"\n')
         deployFile:write('  type: "Pooling"\n')
         -- TOP/BOTTOM
         deployFile:write('  bottom: "' .. (bottom_names[i]) .. '"\n')
         deployFile:write('  top: "' .. top_names[i] .. '"\n')
         -- pooling_param {
         deployFile:write('  pooling_param {\n')
         deployFile:write('    pool: MAX\n')
         -- %%% Pad
         deployFile:write('    pad_h: ' .. tostring(padH) .. '\n')
         deployFile:write('    pad_w: ' .. tostring(padW) .. '\n')
         deployFile:write('    kernel_h: ' .. tostring(nnCur.kH) .. '\n')
         deployFile:write('    kernel_w: ' .. tostring(nnCur.kW) .. '\n')
         deployFile:write('    stride_h: ' .. tostring(nnCur.dH) .. '\n')
         deployFile:write('    stride_w: ' .. tostring(nnCur.dW) .. '\n')
         -- specify ceil_mode
         if (not (nnCur.ceil_mode == true)) then
            deployFile:write('  ceil_mode: false\n')
         end
         deployFile:write('  }\n')
         -- pooling_param }
         -- Update output size
         tempInput = curInput
         curInput[2] = nnCur.nOutputPlane
         curInput[3] = torch.ceil((tempInput[3] - nnCur.kH)/nnCur.dH + 1) -- height
         curInput[4] = torch.ceil((tempInput[4] - nnCur.kW)/nnCur.dW + 1) -- width
      end
      -- Relu
      if (torch.type(nnCur) == 'nn.ReLU') then
         deployFile:write('  name: "' .. names[i] .. '"\n')
         deployFile:write('  type: "ReLU"\n')
         -- TOP/BOTTOM
         deployFile:write('  bottom: "' .. bottom_names[i] .. '"\n')
         deployFile:write('  top: "' .. top_names[i] .. '"\n')
      end
      -- View
      if (torch.type(nnCur) == 'nn.View') then
         deployFile:write('  name: "' .. names[i] .. '"\n')
         deployFile:write('  type: "Reshape"\n')
         -- TOP/BOTTOM
         deployFile:write('  bottom: "' .. bottom_names[i] .. '"\n')
         deployFile:write('  top: "' .. top_names[i] .. '"\n')
         -- reshape_param {
         deployFile:write('  reshape_param {\n')
         -- shape {
         deployFile:write('    shape {\n')
         curInput = {1, 1, 1, 1}
         for isize = 1, #nnCur.size do
            deployFile:write('      dim: ' .. tostring(nnCur.size[isize]) .. '\n')
            curInput[4-#nnCur.size+isize] = nnCur.size[isize]
         end
         deployFile:write('    }\n')
         -- shape }
         deployFile:write('  }\n')
         -- reshape_param }
      end
      --[[
      -- Tanh
      if (torch.type(nnCur) == 'nn.Tanh') then
         deployFile:write('  name: "' .. name[i] .. '"\n')
         deployFile:write('  type: "TANH"\n')
         -- TOP/BOTTOM
         deployFile:write('  bottom: "' .. .. '"\n')
         deployFile:write('  top: "' .. .. '"\n')
      end
      ]]
      --[[
      -- Sigmoid
      if (torch.type(nnCur) == 'nn.Sigmoid') then
         deployFile:write('  name: "' .. name[i] .. '"\n')
         deployFile:write('  type: "ReLU"\n')
         -- TOP/BOTTOM
         deployFile:write('  bottom: "' .. .. '"\n')
         deployFile:write('  top: "' .. .. '"\n')
      end
      ]]
      --[[
      -- %%% LRN
      if (torch.type(nnCur) == 'inn.SpatialCrossResponseNormalization') then
      end
      ]]
      -- Inner_product
      if (torch.type(nnCur) == 'nn.Linear') then
         deployFile:write('  name: "' .. names[i] .. '"\n')
         deployFile:write('  type: "InnerProduct"\n')
         -- dimensions
         innerproductSize = #nnCur:parameters()[1]
         -- TOP/BOTTOM
         deployFile:write('  bottom: "' .. bottom_names[i] .. '"\n')
         deployFile:write('  top: "' .. top_names[i] .. '"\n')
         --[[
         -- blobs {
         deployFile:write('  blobs {\n')
         deployFile:write('    width: ' .. tostring(innerproductSize[2]) .. '\n')
         deployFile:write('  }\n')
         -- blobs }
         --]]
         -- inner_product_param {
         deployFile:write('  inner_product_param {\n')
         deployFile:write('    num_output: ' .. tostring(innerproductSize[1]) .. '\n')
         deployFile:write('    axis: 0\n')
         deployFile:write('  }\n')
         -- inner_product_param }
         -- weights & bias
         paramsFile:write('weights/' .. names[i], nnCur:parameters()[1])
         paramsFile:write('bias/' .. names[i], nnCur:parameters()[2])
         -- update output size
         curInput = {1, 1, 1, innerproductSize[1]}
      end
      -- Dropout
      if (torch.type(nnCur) == 'nn.Dropout') then
         deployFile:write('  name: "' .. names[i] .. '"\n')
         deployFile:write('  type: "Dropout"\n')
         -- TOP/BOTTOM
         deployFile:write('  bottom: "' .. bottom_names[i] .. '"\n')
         deployFile:write('  top: "' .. top_names[i] .. '"\n')
         -- dropout_param {
         deployFile:write('  dropout_param {\n')
         deployFile:write('    dropout_ratio: ' .. tostring(nnCur.p) .. '\n')
         deployFile:write('  }\n')
         -- dropout_param }
      end
      -- Softmax_loss / Softmax
      if (torch.type(nnCur) == 'nn.SoftMax') then
         deployFile:write('  name: "' .. names[i] .. '"\n')
         deployFile:write('  type: "Softmax"\n')
         -- TOP/BOTTOM
         deployFile:write('  bottom: "' .. bottom_names[i] .. '"\n')
         deployFile:write('  top: "' .. top_names[i] .. '"\n')
         -- softmax_param {
         deployFile:write('  softmax_param {\n')
         deployFile:write('    axis: 0\n')
         deployFile:write('  }\n')
         -- }
      end
      deployFile:write('}\n')
      -- layer }
   end
   -- file close
   deployFile:close()
   paramsFile:close()
   print('--- Prototxt file generated.')
end

-- Build Caffe model in Python and output binary weight file
caffeInPy = function(loc, caffeLoc)
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
--w          (default 231)          Width of input image
--h          (default 231)          Height of input image
--loc        (default ./test)       Location to save the outputs
--caffe      (default /opt/caffe)   Location of caffe source
]])

Convert(opt.nf, opt.name, opt.c, opt.w, opt.h, opt.loc, opt.caffe)
