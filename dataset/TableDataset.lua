require 'fn'
require 'fn/seq'
require 'util/arg'
require 'dataset'
require 'dataset/pipeline'
require 'dataset/whitening'
require 'pprint'

local arg = util.arg

local TableDataset = torch.class("dataset.TableDataset")


-- Wraps a table containing a dataset to make it easy to transform the dataset
-- and then sample from.  Each property in the data table must have a tensor or
-- table value, and then each sample will be retrieved by indexing these values
-- and returning a single instance from each one.
--
-- e.g.
--
--   -- a 'dataset' of random samples with random class labels
--   data_table = {
--     data  = torch.Tensor(10, 20, 20),
--     classes = torch.randperm(10)
--   }
--   metadata = { name = 'random', classes = {1,2,3,4,5,6,7,8,9,10} }
--   dataset = TableDataset(data_table, metadata)
--
function TableDataset:__init(data_table, global_metadata)

   self.dataset = data_table

   global_metadata = global_metadata or {}

   self._name = global_metadata.name
   self._classes = global_metadata.classes or {}
end


-- Returns the number of samples in the dataset.
function TableDataset:size()
   return self.dataset.data:size(1)
end


-- Returns the dimensions of a single sample as a table.
-- e.g.
--   mnist          => {1, 28, 28}
--   natural images => {3, 64, 64}
function TableDataset:dimensions()
   local dims = self.dataset.data:size():totable()
   table.remove(dims, 1)
   return dims
end


-- Returns the total number of dimensions of a sample.
-- e.g.
--   mnist => 1*28*28 => 784
function TableDataset:n_dimensions()
   return fn.reduce(fn.mul, 1, self:dimensions())
end


-- Returns the classes represented in this dataset (if available).
function TableDataset:classes()
   return self._classes
end


-- Returns the string name of this dataset.
function TableDataset:name()
   return self._name
end


-- Returns the specified sample (a table) by index.
--
--   sample = dataset:sample(100)
function TableDataset:sample(i)
    local sample = {}

    for key, v in pairs(self.dataset) do
        sample[key] = v[i]
    end

    return sample
end


local function animate(options, samples)
   local rotation    = options.rotation or {}
   local translation = options.translation or {}
   local zoom        = options.zoom or {}
   local frames      = options.frames or 10

   local scratch_a = torch.Tensor()
   local scratch_b = torch.Tensor()

   local function animate_sample(sample)
      local transformers = {}
      if (#rotation > 0) then
         local rot_start, rot_finish = dataset.rand_pair(rotation[1], rotation[2])
         rot_start = rot_start * math.pi / 180
         rot_finish = rot_finish * math.pi / 180
         local rot_delta = (rot_finish - rot_start) / frames
         table.insert(transformers, dataset.rotator(rot_start, rot_delta))
      end

      if (#translation > 0) then
         local xmin_tx, xmax_tx = dataset.rand_pair(translation[1], translation[2])
         local ymin_tx, ymax_tx = dataset.rand_pair(translation[3], translation[4])
         local dx = (xmax_tx - xmin_tx) / frames
         local dy = (ymax_tx - ymin_tx) / frames
         table.insert(transformers, dataset.translator(xmin_tx, ymin_tx, dx, dy))
      end

      if (#zoom > 0) then
         local zoom_start, zoom_finish = dataset.rand_pair(zoom[1], zoom[2])
         local zoom_delta = (zoom_finish - zoom_start) / frames
         table.insert(transformers, dataset.zoomer(zoom_start, zoom_delta))
      end

      local original = sample.data
      scratch_a:resizeAs(sample.data)
      scratch_b:resizeAs(sample.data)
      return seq.repeatedly(frames,
         function()
            scratch_a:zero()
            local a = original
            local b = scratch_b
            for _, transform in ipairs(transformers) do
               transform(a, b)
               a = b
               if a == scratch_a then
                  b = scratch_b
               else
                  b = scratch_a
               end
            end
            sample.data = a
            return sample
         end)
   end

   return seq.mapcat(animate_sample, samples)
end


-- Returns an infinite sequence of data samples.  By default they
-- are shuffled samples, but you can turn shuffling off.
--
--   for sample in seq.take(1000, dataset:sampler()) do
--     net:forward(sample.data)
--   end
--
--   -- turn off shuffling
--   sampler = dataset:sampler({shuffled = false})
--
--   -- generate animations over 10 frames for each sample, which will
--   -- randomly rotate, translate, and/or zoom within the ranges passed.
--   local anim_options = {
--      frames      = 10,
--      rotation    = {-20, 20},
--      translation = {-5, 5, -5, 5},
--      zoom        = {0.6, 1.4}
--   }
--   s = dataset:sampler({animate = anim_options})
--
--   -- pass a custom pipeline for post-processing samples
--   s = dataset:sampler({pipeline = my_pipeline})
--
function TableDataset:sampler(options)
   options = options or {}
   local shuffled = arg.optional(options, 'shuffled', true)
   local indices
   local size = self:size()

   local pipeline, pipe_size = pipe.construct_pipeline(options)

   local function make_sampler()
       if shuffled then
           indices = torch.randperm(size)
       else
           indices = seq.range(size)
       end

       local sample_seq = seq.map(fn.partial(self.sample, self), indices)

       if options.animate then
          sample_seq = animate(options.animate, sample_seq)
       end

       if pipe_size > 0 then
          sample_seq = seq.map(pipeline, sample_seq)
       end

       if options.pipeline then
          sample_seq = seq.map(options.pipeline, sample_seq)
       end

       return sample_seq
    end

   return seq.flatten(seq.cycle(seq.repeatedly(make_sampler)))
end


-- Returns the mini batch starting at the i-th example.
-- Use options.size to specify the mini batch size.
--
--   local batch = dataset:mini_batch(1)
--
--   -- or use directly
--   net:forward(dataset:mini_batch(1).data)
--
--   -- set the batch size using an options table
--   local batch = dataset:mini_batch(1, {size = 100})
--
--   -- or get batch as a sequence of samples, rather than a full tensor
--   for sample in dataset:mini_batch(1, {sequence = true}) do
--     net:forward(sample.data)
--   end
function TableDataset:mini_batch(i, options)
   options = options or {}
   local batch_size   = arg.optional(options, 'size', 10)
   local as_seq = arg.optional(options, 'sequence', false)
   local batch = {}

   if as_seq then
      return seq.map(fn.partial(self.sample, self), seq.range(i, i + batch_size-1))
   else
       for key, v in pairs(self.dataset) do
           batch[key] = v:narrow(1, i, batch_size)
       end

       return batch
   end
end


-- Returns a random mini batch consisting of a table of tensors.
--
--   local batch = dataset:random_mini_batch()
--
--   -- or use directly
--   net:forward(dataset:random_mini_batch().data)
--
--   -- set the batch size using an options table
--   local batch = dataset:random_mini_batch({size = 100})
--
--   -- or get batch as a sequence of samples, rather than a full tensor
--   for sample in dataset:random_mini_batch({sequence = true}) do
--     net:forward(sample.data)
--   end
function TableDataset:random_mini_batch(options)

   options = options or {}
   local batch_size   = arg.optional(options, 'size', 10)
   -- sequence option handled in TableDataset:mini_batch

   return self:mini_batch(torch.random(1, self:size() - batch_size + 1), options)
end


-- Returns a finite sequence of mini batches.
-- The sequence provides each non-overlapping mini batch once.
--
--   -- default options returns contiguous tensors of batch size 10
--   for batch in dataset:mini_batches() do
--      net:forward(batch.data)
--   end
--
--   -- It's also possible to set the size, and/or get the batch as a sequence of
--   -- individual samples.
--   for batch in (seq.take(N_BATCHES, dataset:mini_batches({size = 100, sequence=true})) do
--     for sample in batch do
--       net:forward(sample.data)
--     end
--   end
--
function TableDataset:mini_batches(options)
   options = options or {}
   local shuffled = arg.optional(options, 'shuffled', true)
   local mb_size = arg.optional(options, 'size', 10)
   local indices
   local size = self:size()

   if shuffled then
      indices = torch.randperm(size / mb_size)
   else
      indices = seq.range(size / mb_size)
   end

   return seq.map(function(i)
                     return self:mini_batch((i-1) * mb_size + 1, options)
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
function TableDataset:animation(i)
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
function TableDataset:animations(options)
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


-- Return a pipeline source (i.e. a sequence of samples).
function TableDataset:pipeline_source()
   return self:sampler({shuffled = false})
end


local function channels(...)
   channels = {...}
   if #channels == 0 then
      for i = 1,self.dataset.data:size(2) do
         table.insert(channels, i)
      end
   end
   return channels
end


-- Binarize the dataset: set to 0 any pixel strictly below the threshold, set to 1  those 
-- above or equal to the threshold.
--
-- The argument specifies the threshold value for 0.
function TableDataset:binarize(threshold)

   local function binarize(x, threshold)
       x[x:lt(threshold)] = 0;
       x[x:ge(threshold)] = 1;
       return x
   end

   binarize(self.dataset.data, threshold)
end


-- Globally normalise the dataset (subtract mean and divide by std)
--
-- The optional arguments specify the indices of the channels that should be
-- normalized. If no channels are specified normalize across all channels.
function TableDataset:normalize_globally(...)

   local function normalize(d)
      local mean = d:mean()
      local std = d:std()
      d:add(-mean):div(std)
   end

   local channels = {...}
   if #channels == 0 then
      dataset.normalize(self.dataset.data)
   else
      for _,c in ipairs(channels) do
         normalize(self.dataset.data[{ {}, c, {}, {} }])
      end
   end
end


-- Apply ZCA whitening to dataset (one or more channels)
--
-- The optional arguments specify the indices of the channels that should be
-- normalized. If no channels are specified all channels are jointly whitened.
function TableDataset:zca_whiten(...)
   local channels = {...}
   local P = {}
   local invP = {}
   if #channels == 0 then
      self.dataset.data, P, invP = dataset.zca_whiten(self.dataset.data)
   else
      for _,c in ipairs(channels) do
         self.dataset.data[{ {}, c, {}, {} }], P[c], invP[c] = dataset.zca_whiten(self.dataset.data[{ {}, c, {}, {} }])
      end
   end
   return P, invP
end
