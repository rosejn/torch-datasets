require 'fs'
require 'paths'
require 'image'
require 'pprint'

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
--   path = 'obj2.png',
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

function pipe.pipeline(src, ...)
   local args = {...}
   return function()
      local next_elem = src()
      if next_elem then
         return fn.thread(src(), unpack(args))
      else
         return nil
      end
   end
end


-- Returns a sequence of file names located in dir.
function pipe.file_seq(dir)
  return seq.seq(fs.readdir(dir))
end


-- Returns a list of files located in dir for which string.find will match
-- the expression ex.
local function matching_file_seq(dir, ex)
   return seq.filter(function(name) return string.find(name, ex) end,
                     pipe.file_seq(dir))
end


local function path_seq(dir, files)
   return seq.map(function(filename)
      return {
         filename = filename,
         path = paths.concat(dir, filename)
      }
   end,
   files)
end


-- Returns a sequence of tables with the path property for all files in dir
-- with a matching suffix.  (e.g. { path = 'image_1.png' } )
function pipe.matching_paths(dir, suffix)
   local files = matching_file_seq(dir, suffix)
   return path_seq(dir, files)
end


function pipe.image_paths(dir)
   local files = pipe.file_seq(dir)

   local images = seq.filter(function(path)
      local filename = paths.basename(path)
      local ext = string.match(filename, '%.(%a+)$')
      return image.is_supported(ext)
   end,
   files)

   return path_seq(dir, images)
end


-- Reads sample.path and loads an image (torch.Tensor) into sample.data.
-- RGB images will will be (3 x WIDTH x HEIGHT) dimensions.
-- TODO: might need additional options here for specifying channels etc...
function pipe.image_loader(sample)
   if sample == nil then
      return nil
   end

   local data = image.load(sample.path)
   local dims = data:size()
   sample.width = dims[2]
   sample.height = dims[3]
   sample.data = data
   return sample
end


-- Converts the RGB tensor sample.data to a YUV tensor, separating luminance
-- information from color.
function pipe.rgb2yuv(sample)
   if sample == nil then
      return nil
   end

   sample.data = image.rgb2yuv(sample.data)
   return sample
end


-- Scales the sample.data to be the target width and height.
function pipe.scaler(width, height)
   return function(sample)
      if sample == nil then
         return nil
      end

      sample.data   = image.scale(sample.data, width, height)
      sample.width  = width
      sample.height = height
      return sample
   end
end


-- Crop sample.data to the rect defined by (x1, y1), (x2, y2)
function pipe.cropper(x1, y1, x2, y2)
   return function(sample)
      if sample == nil then
         return nil
      end

      sample.data = image.crop(sample.data, x1, y1, x2, y2)
      return sample
   end
end


-- Crop sample.data to a random patch of size width x height.
function pipe.patch_sampler(width, height)
   return function(sample)
      if sample == nil then
         return nil
      end

      local x = math.random(1, sample.width - width)
      local y = math.random(1, sample.height - height)
      sample.data   = image.crop(sample.data, x, y, x + width, y + height)
      sample.width  = width
      sample.height = height
      return sample
   end
end


-- Using a gaussian kernel, locally subtract the mean and divide by standard
-- deviation.  (Highlight edges and remove areas of low frequency...)
-- e.g.
--   normalizer = pipe.spatial_normalizer(1, 7)
--   sample = normalizer(sample)
function pipe.spatial_normalizer(channel, radius, threshold, the)
   local neighborhood  = image.gaussian1D(radius)
   local normalizer    = nn.SpatialContrastiveNormalization(channel, neighborhood):float()

   return function(sample)
      if sample == nil then
         return nil
      end

      sample.data = normalizer:forward(sample.data[{{channel},{},{}}]:float())
      return sample
   end
end

--[[
function pipe.normalizer(sample)
   local mean = sample.data:mean()
   local std  = 
--]]


function pipe.movie_player(src, fps)
   local movie_win

   for sample in src do
      movie_win = image.display({image = sample.data, win=movie_win, zoom=10})
      util.sleep(1.0 / fps)
   end
end


local display_win
function pipe.display(sample)
   if sample == nil then
      return nil
   end

   display_win = image.display({image = sample.data, win=display_win, zoom=10})

   return sample
end


function pipe.pprint(sample)
   if sample == nil then
      return nil
   end

   pprint(sample)
   return sample
end


local function disk_file_paths(base_path)
   local fname = paths.concat(base_path, '.dat')
   local md_name = paths.concat(base_path, '.md')
   return fname, md_name
end

function pipe.write_to_disk(path, metadata)
   metadata = metadata or {}
   local fname = paths.concat(path, '.dat')
   local md_name = paths.concat(path, '.md')

   local f = torch.DiskFile(fname, 'w')
   f:binary()

   local count = 0
   return function(sample)
      if sample == nil then
         f:close()

         local md_file = torch.DiskFile(md_name, 'w')
         md.size = count
         md_file:writeObject(md)
         md_file:close()

         return nil
      end

      f:writeObject(sample)
      count = count + 1
      return sample
   end
end


function pipe.load_from_disk(path)
    local f = torch.DiskFile(path, 'r')
    f:binary()

    local n_examples   = f:readInt()
    local n_dimensions = f:readInt()

    if n then
        n_examples = n
    end
    local tensor       = torch.Tensor(n_examples, n_dimensions)
    tensor:storage():copy(f:readFloat(n_examples * n_dimensions))

    return n_examples, n_dimensions, tensor
end
--]]
