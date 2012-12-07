require 'torch'
require 'fs'
require 'xlua'
require 'logroll'

local log = logroll.print_logger()

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



local function data(files, opt)

	-- TODO: either combine pairs into one, keep both, keep left, keep right
	-- TODO: grab subset for each class
	-- TODO: have a standard option to get only 'n' images

	local raw = util.merge(
		split_metadata(raw_data(files.info)),
		{
			data    = raw_data(files.dat):double(),
			classes = raw_data(files.cat)
		}
	)
	local stages = {
		pipe.data_table_source(raw),
		pipe.div(256) -- always normalise to range [0, 1]
	}

	local pipeline = pipe.pipeline(unpack(stages))
	local table = pipe.to_data_table(SmallNorb.size/2, pipeline)
	
	return dataset.TableDataset(table)
end



function SmallNorb.test_data(opt)
	return data(SmallNorb.testing_files, opt)
end


function SmallNorb.train_data(opt)
	return data(SmallNorb.training_files, opt)
end


return SmallNorb
