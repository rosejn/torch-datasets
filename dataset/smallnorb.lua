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


SmallNorb = {}

SmallNorb.name           = 'smallnorb'
SmallNorb.dimensions     = {1, 96, 96}
SmallNorb.n_dimensions   = 1 * 96 * 96
SmallNorb.size           = 2 * 24300

SmallNorb.classes        = {'animal', 'human', 'airplane', 'truck', 'car', 'none'}

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



--[[ Add standard pipeline stages

Optional parameters:

* downsample : downsample by some constant factor
* normalise

--]]
function standard_options(opts, stages)

	local downsample = arg.optional(opts, 'downsample', 0)
	local normalize  = arg.optional(opts, 'normalize', false)

	if normalize then
		table.insert(stages, pipe.normalize)
	end

	if downsample then
		local width = SmallNorb.dimensions[3] / downsample
		local height = SmallNorb.dimensions[2] / downsample
		table.insert(stages, pipe.scaler(width, height))
	end

	return stages
end



local function process_pairs(pair_format, stages)

	local function half(n)
		return function(sample) sample.data = sample.data[n] return sample end
	end

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


--[[

Parameters:

* n_frames (unsigned) : number of frames to return
* pairs (optional string : ('combined' | 'left' | 'right'):
	How should stereo pairs be loaded? 'combined' returns the two sub-images in each example,
	'left' and 'right' return only one half of the images.

--]]
local function data(files, opt)

	opt = opt or {}

	local n_frames = arg.optional(opt, 'n_frames', SmallNorb.size/2)
	local pair_format = arg.optional(opt, 'pairs', 'combined')

	-- TODO: grab subset for each class

	local raw = util.merge(
		split_metadata(raw_data(files.info)),
		{
			data    = raw_data(files.dat):float(),
			classes = raw_data(files.cat)
		}
	)

	collectgarbage()

	local stages = standard_options(opt,
		process_pairs(pair_format, {
			pipe.data_table_source(raw),
			pipe.div(256) -- always normalise to range [0, 1]
		})
	)

	local pipeline = pipe.pipeline(unpack(stages))
	local table = pipe.to_data_table(n_frames, pipeline)
	
	return dataset.TableDataset(table)
end



function SmallNorb.test_data(opt)
	return data(SmallNorb.testing_files, opt)
end


function SmallNorb.train_data(opt)
	return data(SmallNorb.training_files, opt)
end


return SmallNorb
