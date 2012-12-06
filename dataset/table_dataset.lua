require 'dataset'
require 'fn'
require 'fn/seq'
require 'util/arg'
local arg = util.arg

local TableDataset = torch.class("dataset.TableDataset")

function TableDataset:__init(data_table)
    self.dataset = data_table

    --util.set_index_fn(self, self.sample)
    --util.set_size_fn(self, self.size)
end


function TableDataset:size()
   return self.dataset.data:size(1)
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


-- Returns an infinite sequence of data samples.  By default they
-- are shuffled samples, but you can turn shuffling off.
--
--   for sample in seq.take(1000, dataset:sampler()) do
--     net:forward(sample.data)
--   end
--
--   -- turn off shuffling
--   sampler = dataset:sampler({shuffled = false})
function TableDataset:sampler(options)
   options = options or {}
   local shuffled = arg.optional(options, 'shuffled', true)
   local indices
   local size = self:size()

   local function make_sampler()
       if shuffled then
           indices = torch.randperm(size)
       else
           indices = seq.range(size)
       end
       return seq.map(fn.partial(self.sample, self), indices)
   end

   return seq.flatten(seq.cycle(seq.repeatedly(make_sampler)))
end


-- Returns the ith mini batch consisting of a table of tensors.
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


-- Returns an infinite sequence of mini batches.
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

--[[
* get sampled animation sequences
 - add frame # to each sample

--]]
