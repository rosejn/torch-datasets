require 'torch'
require 'image'
require 'paths'
require 'dok'

require 'fn'
require 'fn/seq'
require 'util'
require 'util/file'
require 'util/arg'
local arg = util.arg

require 'dataset'

local Mnist = torch.class("dataset.Mnist")
Mnist.name         = 'mnist'
Mnist.dimensions   = {1, 28, 28}
Mnist.n_dimensions = 1 * 28 * 28
Mnist.size         = 60000
Mnist.classes      = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
Mnist.url          = 'http://data.neuflow.org/data/mnist-th7.tgz'
Mnist.file         = 'mnist-th7/train.th7'

Mnist.test_file    = 'mnist-th7/test.th7'
Mnist.test_size    = 10000

-- Get the raw, unprocessed dataset.
-- Returns a 60,000 x 785 tensor, where each image is 28*28 = 784 values in the
-- range [0-255], and the 785th element is the class ID.
function Mnist.raw_data(n)
   local path = dataset.data_path(Mnist.name, Mnist.url, Mnist.file)
   local n_examples, n_dimensions, data = dataset.load_data_file(path, n)
   return data
end

-- Get the raw, unprocessed test dataset.
-- Returns a 10,000 x 785 tensor in the same format as the training set
-- described above.
function Mnist.raw_test_data(n)
   local path = dataset.data_path(Mnist.name, Mnist.url, Mnist.test_file)
   local n_examples, n_dimensions, data = dataset.load_data_file(path, n)
   return data
end

-- Setup an MNIST dataset instance.
--
--   m = dataset.Mnist()
--
--   -- scale values between [0,1] (by default they are in the range [0,255])
--   m = dataset.Mnist({scale = {0, 1}})
--
--   -- or normalize (subtract mean and divide by std)
--   m = dataset.Mnist({normalize = true})
--
--   -- only import a subset of the data (imports full 60,000 samples otherwise)
--   m = dataset.Mnist({size = 1000})
--
--   -- optionally animate mnist digits using translation, rotation, and
--   -- scaling over a certain number of frames.
--   m = dataset.Mnist{frames = frames,
--                     rotation = {-20, 20},
--                     zoom = {0.3, 1.5},
--                     translation = {-8, 8, -8, 8}
--                    }
--
--   -- use the test data rather than the training data:
--   m = dataset.Mnist({test = true})
function Mnist:__init(opts)
   local animated, animated_labels
   local scale, normalize, size, frames, rotation, translation, zoom
   --[[ TODO: dok.unpack seems broken...

   local _, scale, normalize, size, frames, rotation, translation, zoom = dok.unpack({...},
         'Mnist:__init',
         'returns a pre-processed MNIST dataset',
         {arg='scale',       type='table',   help='scale dataset within a range {min, max}', default={}},
         {arg='normalize',   type='boolean', help='apply global normalization => (data - mean) / std', default=false},
         {arg='size',        type='number',  help='specify a size if you only want a subset of the data', default=Mnist.size},
         {arg='frames',      type='number',  help='specify the number of frames to animate each sample', default=10},
         {arg='rotation',    type='table',   help='rotation parameters = {min, max}', default=nil},
         {arg='translation', type='table',   help='translation parameters = {xmin, xmax, ymin, ymax}', default=nil},
         {arg='zoom',        type='table',   help='scaling parameters = {min, max}', default=nil})
         ]]
   opts        = opts or {}
   test        = arg.optional(opts, 'test', false)
   scale       = arg.optional(opts, 'scale', {})
   normalize   = arg.optional(opts, 'normalize', false)
   size        = arg.optional(opts, 'size', test and Mnist.test_size or Mnist.size)
   frames      = arg.optional(opts, 'frames', 10)
   rotation    = arg.optional(opts, 'rotation', {})
   translation = arg.optional(opts, 'translation', {})
   zoom        = arg.optional(opts, 'zoom', {})
   sorted      = arg.optional(opts, 'sort', false)

   local data
   if test then
      data = Mnist.raw_test_data(size)
   else
      data = Mnist.raw_data(size)
   end
   local samples = data:narrow(2, 1, Mnist.n_dimensions):clone()

   local labels = torch.Tensor(size)
   for i=1,size do
      labels[i] = data[{i, 785}]
   end

   if normalize then
       mean, std = dataset.global_normalization(samples)
   end

   if (#scale > 0) then
      dataset.scale(samples, scale[1], scale[2])
   end

   if sorted then
       samples, labels = dataset.sort_by_class(samples, labels)
   end

   self.samples   = samples
   self.labels    = labels
   self.size      = size
   self.frames    = frames
   self.label_vector = torch.zeros(#Mnist.classes)

   if (#rotation > 0) or (#translation > 0) or (#zoom > 0) then
      self:_animate(rotation, translation, zoom)
   end
end


function Mnist:_animate(rotation, translation, zoom)
   local full_size = self.frames * self.size
   animated = torch.Tensor(full_size, Mnist.n_dimensions):zero()
   animated_labels = torch.Tensor(full_size)

   for i=1,self.size do
      for f=1,self.frames do
         animated_labels[i+f] = self.labels[i]
      end
   end

   for sample=1,self.size do
      local transformers = {}
      if (#rotation > 0) then
         local rot_start, rot_finish = dataset.rand_pair(rotation[1], rotation[2])
         rot_start = rot_start * math.pi / 180
         rot_finish = rot_finish * math.pi / 180
         local rot_delta = (rot_finish - rot_start) / self.frames
         table.insert(transformers, dataset.rotator(rot_start, rot_delta))
         self.rotation = rotation
      end

      if (#translation > 0) then
         local xmin_tx, xmax_tx = dataset.rand_pair(translation[1], translation[2])
         local ymin_tx, ymax_tx = dataset.rand_pair(translation[3], translation[4])
         local dx = (xmax_tx - xmin_tx) / frames
         local dy = (ymax_tx - ymin_tx) / frames
         table.insert(transformers, dataset.translator(xmin_tx, ymin_tx, dx, dy))
         self.translation = translation
      end

      if (#zoom > 0) then
         local zoom_start, zoom_finish = dataset.rand_pair(zoom[1], zoom[2])
         local zoom_delta = (zoom_finish - zoom_start) / self.frames
         table.insert(transformers, dataset.zoomer(zoom_start, zoom_delta))
         self.zoom = zoom
      end

      local src = self.samples[sample]:unfold(1, 28, 28)
      for f=1,self.frames do
         local dst = animated:narrow(1, (sample-1)*self.frames + f, 1):select(1,1):unfold(1, 28, 28)
         local tsrc = src
         for _, transform in ipairs(transformers) do
            transform(tsrc, dst)
            tsrc = dst
         end
      end
   end

   self.samples      = animated
   self.labels    = animated_labels
   self.base_size = self.size
   self.size      = full_size
end

-- Returns the specified (sample, label) pair.
--
--   sample, label = m:sample(100)
function Mnist:sample(i)
   return self.samples[i]:double(), self.labels[i]
end

-- Returns a sequence of shuffled (sample, label) pairs.
--
--   for sample, label in m:samples() do
--     net:forward(sample)
--   end
--
--   -- you can optionally turn off shuffling
--   sample_seq = m:sampler({shuffled = false})
function Mnist:sampler(options)
   options = options or {}
   local shuffled = arg.optional(options, 'shuffled', true)
   local indices

   if shuffled then
      indices = torch.randperm(self.size)
   else
      indices = seq.range(self.size)
   end

   return seq.map(fn.partial(self.sample, self), indices)
end


-- Returns the ith mini batch tuple consisting of (batch_tensor, labels_tensor) pair.
--
--   local batch, labels = m:mini_batch(1)
--
--   -- or use directly
--   net:forward(m:mini_batch(1))
--
--   -- set the batch size using an options table
--   local batch, labels = m:mini_batch(1, {size = 100})
--
--   -- or get batch as a sequence of samples, rather than a full tensor
--   for sample, label in m:mini_batch(1, {sequence = true}) do
--     net:forward(sample)
--   end
function Mnist:mini_batch(i, options)
   options = options or {}
   local size   = arg.optional(options, 'size', 10)
   local as_seq = arg.optional(options, 'sequence', false)

   if as_seq then
      return seq.map(fn.partial(self.sample, self), seq.range(i, i+size-1))
   else
      local batch  = self.samples:narrow(1, i, size)
      local labels = self.labels:narrow(1, i, size)
      return batch, labels
   end
end

-- Returns a sequence of mini batches.
--
--   -- default options returns contiguous tensors of batch size 10
--   for batch, labels in m:mini_batches() do
--      net:forward(batch)
--   end
--
--   -- also possible to set the size, and/or get the batch as a sequence of
--   -- individual samples.
--   for batch in m:mini_batches({size = 100, sequence=true}) do
--     for sample,label in batch do
--       net:forward(sample)
--     end
--   end
--
function Mnist:mini_batches(options)
   options = options or {}
   local shuffled = arg.optional(options, 'shuffled', true)
   local mb_size = arg.optional(options, 'size', 10)
   local indices

   if shuffled then
      indices = torch.randperm(self.size / mb_size)
   else
      indices = seq.range(self.size / mb_size)
   end

   return seq.map(function(i)
                     return self:mini_batch((i-1)*mb_size+1, options)
                  end,
                  indices)
end

-- Returns the sequence of frames corresponding to a specific sample's animation.
--
--   for frame,label in m:animation(1) do
--      local img = frame:unfold(1,28,28)
--      win = image.display({win=win, image=img, zoom=10})
--      util.sleep(1 / 24)
--   end
--
function Mnist:animation(i)
   local start = ((i-1) * self.frames) + 1
   return self:mini_batch(start, self.frames, {sequence = true})
end


-- Returns a sequence of animations, where each animation is a sequence of
-- samples.
--
--   for anim in m:animations() do
--      for frame,label in anim do
--         local img = frame:unfold(1,28,28)
--         win = image.display({win=win, image=img, zoom=10})
--         util.sleep(1 / 24)
--      end
--   end
--
function Mnist:animations(options)
   options = options or {}
   local shuffled = arg.optional(options, 'shuffled', true)
   local indices

   if shuffled then
      indices = torch.randperm(self.base_size)
   else
      indices = seq.range(self.base_size)
   end
   return seq.map(function(i)
                     return self:animation(i)
                  end,
                  indices)
end

