require 'fs'
require 'image'

require 'util'
require 'fn'
require 'fn/seq'

-- A dataset pipeline system, allowing for easy loading and transforming of
-- datasets.  A pipeline processes tables representing dataset samples:
--
-- { data     = <torch.Tensor <channels x width x height>,
--   class    = class_id,
--
--   -- if loaded from a file
--   filename = 'obj2.png',
--
--   -- animations will contain additional metadata such as:
--   frame    = 9,
--   rotation = 45,
--   zoom     = 0.2
-- }
--
-- If it's an image the data property should always be a tensor which is of the
-- channels by image dimensions, so that it can be displayed and operated on
-- without having to store this information somewhere else.

pipe = {}


-- Returns a sequence of file names located in dir.
local function file_seq(dir)
  return seq.seq(fs.readdir(dir))
end


-- Returns a list of files located in dir for which string.find will match
-- the expression ex.
local function matching_file_seq(dir, ex)
   return seq.filter(function(name) return string.find(name, ex) end,
                     file_seq(dir))
end


local function filename_seq(files)
   return seq.map(function(name)
      return {filename = name}
   end,
   files)
end


-- Returns a sequence of tables with the filename property for all files in dir
-- with a matching suffix.  (e.g. { filename = 'image_1.png' } )
function pipe.filenames(dir, suffix)
   local files = matching_file_seq(dir, suffix)
   return filename_seq(files)
end


function pipe.image_filenames(dir)
   local files = file_seq(dir)

   local images = seq.filter(function(name)
      local ext = string.match(filename,'%.(%a+)$')
      return image.is_supported(ext)
   end,
   files)

   return filename_seq(images)
end


-- Reads info.filename and loads an image (torch.Tensor) into sample.data.
-- RGB images will will be (3 x WIDTH x HEIGHT) dimensions.
function pipe.image_loader(info)
   local data = image.load(paths.concat(dir, info.filename))
   local dims = data:size()
   info.width = dims[2]
   info.height = dims[3]
   info.data = data
   return info
end


-- Converts the RGB tensor sample.data to a YUV tensor, separating luminance
-- information from color.
function pipe.rgb2yuv(sample)
   sample.data = image.rgb2yuv(sample.data)
   return sample
end


-- Scales the sample.data to be the target width and height.
function pipe.scaler(width, height, sample)
   sample.data   = image.scale(sample.data, width, height)
   sample.width  = width
   sample.height = height
   return sample
end


-- Crop sample.data to the rect defined by (x1, y1), (x2, y2)
function pipe.cropper(x1, y1, x2, y2, sample)
   sample.data = image.crop(sample.data, x1, y1, x2, y2)
   return sample
end


-- Crop sample.data to a random patch of size width x height.
function pipe.patch_sampler(width, height, sample)
   local x = math.random(1, sample.width - width)
   local y = math.random(1, sample.height - height)
   sample.data = image.crop(sample.data, x, y, x + width, y + height)
   sample.width = width
   sample.height = height
   return sample
end


-- Using a gaussian kernel, locally subtract the mean and divide by standard
-- deviation.  (Highlight edges and remove areas of low frequency...)
-- e.g.
--   pipe.spatial_normalization(1, 7, sample)
function pipe.spatial_normalization(channel, radius, sample)
   local neighborhood  = image.gaussian1D(radius)
   local normalizer    = nn.SpatialContrastiveNormalization(1, neighborhood):float()
   sample.data = normalizer:forward(sample.data[{{1},{},{}}])
   return sample
end

local movie_win
function pipe.movie_player(fps, sample)
   movie_win = image.display({win=movie_win, image=sample.data, zoom=10})
   util.sleep(1.0 / fps)
end
