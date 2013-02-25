require 'torch'
require 'fs'
require 'xlua'
require 'logroll'

local log = logroll.print_logger()

local arg = require 'util/arg'

local bit = require 'bit'
local lshift, bor = bit.lshift, bit.bor

package.path = '?/init.lua;' .. package.path

require 'dataset'
require 'dataset/pipeline'
require 'dataset/lushio'
require 'dataset/TableDataset'

require 'dataset/whitening'

SmallNorb = {}

SmallNorb.name           = 'smallnorb'
SmallNorb.dimensions     = {1, 96, 96}
SmallNorb.n_dimensions   = 1 * 96 * 96
SmallNorb.size           = 2 * 24300
SmallNorb.classes        = {'animal', 'human', 'airplane', 'truck', 'car'}

-- The small NORB dataset is spread over six zip files from the following location
SmallNorb.url            = 'http://www.cs.nyu.edu/~ylclab/data/norb-v1.0-small/'

SmallNorb.training_files = {
	cat  = 'smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat',
	dat  = 'smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat',
	info = 'smallnorb-5x46789x9x18x6x2x96x96-training-info.mat'
}

SmallNorb.testing_files  = {
	cat  = 'smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat',
	dat  = 'smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat',
	info = 'smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat'
}



--[[ Get a single matrix from NORB in torch format --]]
local function raw_data(filename, toTensor)
	local function unzip(filename) os.execute('gunzip ' .. filename) end
	local bin_matrix = dataset.data_path(SmallNorb.name, SmallNorb.url .. filename .. '.gz', filename, unzip)
	return lushio.read(bin_matrix)
end


--[[ Split metadata into separate fields

The fields are:

* instance in the category (0 to 9)
* the elevation (0 to 8, which mean cameras are 30, 35,40,45,50,55,60,65,70 degrees from the horizontal respectively)
* the azimuth (0,2,4,...,34, multiply by 10 to get the azimuth in degrees)
* the lighting condition (0 to 5)

--]]
local function split_metadata(metadata)
	return {
		instance  = metadata:select(2,1),
		elevation = metadata:select(2,2),
		azimuth   = metadata:select(2,3),
		lighting  = metadata:select(2,4)
	}
end



local function normalize(enabled)
	return function(stages)
		if enabled then
			table.insert(stages, pipe.normalizer)
		end
		return stages
	end
end



local function downsample(factor)
	return function(stages)
		if factor ~= nil then
			local width = SmallNorb.dimensions[3] / factor
			local height = SmallNorb.dimensions[2] / factor
			table.insert(stages, pipe.resizer(width, height))
			return stages
		else
			return stages
		end
	end
end



local function process_pairs(pair_format)

	local function half(n)
		return function(sample) sample.data = sample.data:narrow(1,n,n) return sample end
	end

	return function(stages)
		if pair_format == 'combined' then
			-- The data already comes in combined form
		elseif pair_format == 'left' then
			table.insert(stages, half(1))
		elseif pair_format == 'right' then
			table.insert(stages, half(2))
		else
			error("unknown 'pairs' argument " .. pair_format)
		end
		return stages
	end
end



-- Invert table
local function invert(table)
	local ret = {}
	for k,v in pairs(table) do
		ret[v] = k	
	end
	return ret
end



-- Return a numerical class id from a class description that's either a string or an id 
local function class_id(class)
	if type(class) == 'number' then
		if class < 0 or class > 4 then
			error('invalid NORB class id. 0 =< id =< 4') 
		end
		return class
	else
		return invert(SmallNorb.classes)[class]-1
	end
end


local function filter(val, field, transform)
	transform = transform or fn.id
	return function(source)
		if val then
			local newval = transform(val)
			return pipe.filter(source, field, newval)
		else
			return source
		end
	end
end



--[[

Parameters:

* test (optional boolean, default : false)
	use test set instead of training set
* size (optional unsigned) :
	number of frames to return
* pairs (optional string : ('combined' | 'left' | 'right'):
	How should stereo pairs be loaded? 'combined' returns the two sub-images in each example,
	'left' and 'right' return only one half of the images.
* downsample (optional unsigned >= 1):
	downsample by some constant factor
* normalise (optional bool) :
* class (number or string) :
	restrict dataset to the given object class
* instance (number) :
	restrict dataset to the given object instance
* elevation (number) :
	restrict dataset to objects with the given elevation
* azimuth (number) :
	restrict dataset to objects with the given azimuth
* lighting (number) :
	restrict dataset to objects with the given lighting

--]]
function SmallNorb.dataset(opt)

	opt = opt or {}

	local test              = arg.optional(opt, 'test', false)
	local size              = arg.optional(opt, 'size', SmallNorb.size/2)
	local pair_format       = arg.optional(opt, 'pairs', 'combined')
	local class             = arg.optional(opt, 'class')
	local downsample_factor = arg.optional(opt, 'downsample')
	local do_normalize      = arg.optional(opt, 'normalize', false)
	local zca_whiten        = arg.optional(opt, 'zca_whiten', false)
	local instance          = arg.optional(opt, 'instance')
	local elevation         = arg.optional(opt, 'elevation')
	local azimuth           = arg.optional(opt, 'azimuth')
	local lighting          = arg.optional(opt, 'lighting')

	local files
	if test then
		files = SmallNorb.testing_files
	else
		files = SmallNorb.training_files
	end

	local raw = util.merge(
		split_metadata(raw_data(files.info)),
		{
			data  = raw_data(files.dat):float(),
			class = raw_data(files.cat)
		}
	)

	collectgarbage()

	-- TODO: verify that arguments are valid
	local source = fn.thread(
		dataset.TableDataset(raw):sampler{shuffled = false},
		filter(class, 'classes', class_id),
		filter(instance, 'instance'),
		filter(elevation, 'elevation'),
		filter(azimuth, 'azimuth'),
		filter(lighting, 'lighting')
	)

	local stages = fn.thread({
			source,
			pipe.div(256) -- always normalise to range [0, 1]
		},
		process_pairs(pair_format),
		downsample(downsample_factor),
		normalize(do_normalize)
	)

	local pipeline = pipe.pipeline(unpack(stages))
	local table = pipe.data_table_sink(size, pipeline)
	
	local d =  dataset.TableDataset(table, SmallNorb)

	if zca_whiten then
		d:zca_whiten()
	end

	return d
end



return SmallNorb
