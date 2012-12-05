require 'torch'
require 'fs'
require 'xlua'
require 'logroll'

local log = logroll.print_logger()

local bit = require 'bit'
local lshift, bor = bit.lshift, bit.bor

-- package.path = '?/init.lua;' .. package.path

require 'dataset'
require 'dataset/pipeline'


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



--[[ Convert a binary matrix (in a file) to a Torch tensor --]]
local function binary_matrix_to_torch(filename)

	local f = assert(io.open(filename, 'rb'))

	local function byte(str, pos)
		return string.byte(str, pos)
	end

	local function int32(str, pos)
		local b0, b1, b2, b3 = string.byte(str, pos, pos+4)
		return bor(lshift(b3,24), lshift(b2,16), lshift(b1,8), lshift(b0,0))
	end

	-- Magic numbers etc. described at http://www.cs.nyu.edu/~ylclab/data/norb-v1.0-small
	decoder = {}
	decoder[0x1e3d4c55] = { torch.ByteTensor , byte , 1 }
	decoder[0x1e3d4c54] = { torch.IntTensor  , int32, 4 }

	local header = f:read(4+4+16)
	local magic = int32(header, 1)
	local ndim = int32(header, 1+4)
	local dim = {}

	for i = 1,ndim do
		dim[i] = int32(header, 1+4+i*4) 
	end

	local tensor, convert, wordsize = unpack(decoder[magic])

	local ret = tensor(unpack(dim))
	local header_size = 8 + 4 * math.max(3,ndim)
	f:seek('set', header_size)

	local function read_row(t, n_words)
		local bytes = f:read(n_words * wordsize)
		for i = 1,n_words do
			ret[i] = convert(bytes, 1+(i-1)*wordsize)
		end
	end

	log.info('Converting to Torch format')
	if ndim == 1 then
		read_row(ret, dim[1])
	elseif ndim == 2 then
		for i = 1,dim[1] do
			xlua.progress(i, dim[1])
			read_row(ret[i], dim[2])
		end
	elseif ndim == 4 then
		for i = 1,dim[1] do
			xlua.progress(i, dim[1])
			for j = 1,dim[2] do
				for k = 1,dim[3] do
					read_row(ret[i][j][k], dim[4])
				end
			end
		end
	end

	f:close()
	collectgarbage()

	return ret
end



--[[ Get a single matrix from NORB in torch format --]]
local function raw_data(filename)

	local function unzip(filename) os.execute('gunzip ' .. filename) end

	local local_path = paths.concat(dataset.data_dir, SmallNorb.name, filename .. '.th7')

	if is_file(local_path) then
		return torch.load(local_path)
	else
		local bin_matrix = dataset.data_path(SmallNorb.name, SmallNorb.url .. filename .. '.gz', filename, unzip)
		local th_matrix = binary_matrix_to_torch(bin_matrix)
		torch.save(local_path, th_matrix)
		return th_matrix
	end
end



--[[ Split metadata into separate fields 

The fields are:

* instance in the category (0 to 9)
* the elevation (0 to 8, which mean cameras are 30, 35,40,45,50,55,60,65,70 degrees from the horizontal respectively)
* the azimuth (0,2,4,...,34, multiply by 10 to get the azimuth in degrees)
* the lighting condition (0 to 5)

--]]
function SmallNorb.split_metadata(metadata)
	return {
		instance  = metadata:select(2,1),
		elevation = metadata:select(2,2),
		azimuth   = metadata:select(2,3),
		lighting  = metadata:select(2,4)
	}
end



function SmallNorb.dataset(files)
	return util.merge(
		SmallNorb.split_metadata(raw_data(files.info)), 
		{
			data    = raw_data(files.dat),
			classes = raw_data(files.cat)
		}
	)
end


function SmallNorb.test_data()  return SmallNorb.dataset(SmallNorb.testing_files ) end
function SmallNorb.train_data() return SmallNorb.dataset(SmallNorb.training_files) end


-- TODO: Pipeline:
-- convert to float data and scale appropriately
-- (possibly) resize 

return SmallNorb
