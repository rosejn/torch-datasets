require 'dataset'

local arg = require 'util/arg'

local SimpleDataset = torch.class('dataset.SimpleDataset')

function SimpleDataset:__init(data_table,metadata_table)

	metadata_table = metadata_table or {}

	self.dataset = data_table
	self.metadata = metadata_table

	if not self.dataset.data then
		error('data_table does not contain "data" field')
	end
end

function SimpleDataset:size()
	local data = self.dataset.data
	if type(data) == 'table' then
		return #data
	else
		return data:size(1)
	end
end

-- Returns the specified sample (a table) by index.
--
--   sample = dataset:sample(100)
function SimpleDataset:sample(i)
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
--   my_pipeline = pipe.line({pipe.lcn(),pipe.patch_sampler(20,20)})
--   s = dataset:sampler({pipeline = my_pipeline})
--
function SimpleDataset:sampler(options)
   options = options or {}
   local shuffled = arg.optional(options, 'shuffled', true)
   local indices
   local size = self:size()

   local function make_sampler()
       if shuffled then
           indices = torch.randperm(size)
       else
           indices = torch.range(1,size)
       end

       local sample_seq = seq.map(fn.partial(self.sample, self), indices)

       if options.pipeline then
          sample_seq = seq.map(options.pipeline, sample_seq)
       end

       return sample_seq
    end

   return seq.flatten(seq.cycle(seq.repeatedly(make_sampler)))
end


