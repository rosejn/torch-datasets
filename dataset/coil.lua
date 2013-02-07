require 'torch'
require 'image'
require 'paths'
require 'fs'

require 'util'
require 'util/file'
require 'dataset/pipeline'
require 'fn'
require 'fn/seq'

require 'dataset'
require 'dataset/TableDataset'

Coil = {}

Coil.name         = 'coil'
Coil.dimensions   = {3, 128, 128}
Coil.n_dimensions = 3 * 128 * 128
Coil.size         = 7200
Coil.url          = 'http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip'
Coil.file         = 'coil-100'

Coil.default_options = {
    size      = Coil.size,
    resize    = {64, 64},
    yuv       = true,
    normalize = true,
    whitening = {1, 7, 1, 1},
}

local function coil_image_dir()
	return dataset.data_path(Coil.name, Coil.url, Coil.file)
end


-- Parse a coil file path and return a table of metadata with the image number
-- and angle of the object.
local function coil_metadata_extractor(sample)
   _, _, img, angle = string.find(sample.filename, "obj(%d+)__(%d+).png")
   img   = tonumber(img)

   sample.image = img
   sample.class = img
   sample.angle = tonumber(angle)

   return sample
end


-- Returns a sequence of tables representing the coil images sorted by image number and angle.
local function image_paths(path)
   local files = pipe.file_source(path, 'obj')
   local file_maps = seq.table(seq.map(coil_metadata_extractor, files))
   local numerical_order = function(a, b)
      if a.image < b.image then
         return true
      elseif a.image == b.image then
         return a.angle < b.angle
      else
         return false
      end
   end

   table.sort(file_maps, numerical_order)
   return seq.seq(file_maps)
end


-- Returns a sequence of Coil rgb images samples.
function Coil.data_source(size)
   size = size or Coil.size

   local line = pipe.line({coil_metadata_extractor,
                           pipe.image_loader,
                           pipe.remove_keys('filename', 'path', 'width', 'height'),
                           function(sample)
                              sample.frame = (sample.angle / 5) + 1
                              return sample
                           end,
                       })

   local image_dir = coil_image_dir()
   return seq.take(size, pipe.connect(image_paths(image_dir), line))
end



-- Returns a TableDataset for Coil with each sample of size 3 x width x height,
-- in a normalized YUV color format.
function Coil.dataset(options)
   options = options or Coil.default_options

   local pipeline = pipe.construct_pipeline(options)
   local data     = pipe.connect(Coil.data_source(options.size), pipeline)
   local table    = pipe.data_table_sink(options.size, data)

   return dataset.TableDataset(table, Coil)
end


