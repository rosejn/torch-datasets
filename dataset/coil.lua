require 'torch'
require 'image'
require 'paths'
require 'fs'
require 'nn'

require 'util'
require 'util/arg'
require 'util/file'
require 'dataset/pipeline'
require 'fn'
require 'fn/seq'

local arg = util.arg
--local Coil = torch.class("dataset.Coil")
Coil = {}

Coil.name         = 'coil'
Coil.dimensions   = {1, 128, 128}
Coil.n_dimensions = 3 * 128 * 128
Coil.size         = 7200
Coil.url          = 'http://somewhere.com/coil-100.t7.zip'
Coil.file         = 'coil-100.t7'

function Coil.load_data()
   local path = dataset.data_path(Coil.name, Coil.url, Coil.file)
   local n_objects, n_dimensions, data = dataset.load_data_file(path)
   return data
end

function Coil:raw_rgb_data()
   return n_examples, n_dimensions, data
end

--[[
function Coil:get(object, angle)
   local index = object + (angle / 5)
   local input = data[index]:narrow(1, 1, n_dimensions - 2):double()
   local object_id = data[index][n_dimensions-1]
   local angle = data[index][n_dimensions]

   local display = function()
      image.display{image=input:unfold(unpack(dataset.dimensions)),
      zoom=4, legend='Coil[' .. object .. ',' .. angle ']'}
   end

   return {
      input   = input,
      object  = object_id,
      angle   = angle
      display = display
   }
end

function
   --mean, std = dataset.global_normalization(data:narrow(2, 1, n_dimensions - 2))

   local dataset = util.merge(util.copy(coil_md), {
      data     = data,
      channels = {'r', 'g', 'b'},
      mean     = mean,
      std      = std,
      size     = function() return n_examples end,
      n_dimensions = n_dimensions - 2,
   })

   util.set_index_fn(dataset,
   function(self, index)
      local input = data[index]:narrow(1, 1, n_dimensions - 2):double()
      local object_id = data[index][n_dimensions-1]
      local angle = data[index][n_dimensions]

      local display = function()
         image.display{image=input:unfold(unpack(dataset.dimensions)),
                       zoom=4, legend='mnist[' .. index .. ']'}
      end

      return {
         input   = input,
         object  = object_id,
         angle   = angle
         display = display
      }
   end)

   util.set_size_fn(dataset,
   function(self)
      return self.size()
   end)

   return dataset
end
]]


-- Parse a coil file path and return a table of metadata with the image number
-- and angle of the object.
local function coil_metadata_extractor(sample)
   _, _, img, angle = string.find(sample.filename, "obj(%d+)__(%d+).ppm")
   img   = tonumber(img)

   sample.image = img
   sample.class = img
   sample.angle = tonumber(angle)

   return sample
end

-- Returns a sequence of tables representing the coil images sorted by image number and angle.
function Coil.image_paths(dir)
   local files = pipe.matching_paths(dir, 'obj')
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


function coil_images(dir, width, height)
   return pipe.pipeline(Coil.image_paths(dir),
                        coil_metadata_extractor,
                        pipe.image_loader,
                        pipe.scaler(width, height),
                        pipe.rgb2yuv,
                        pipe.normalizer,
                        pipe.spatial_normalizer(1, 7, 1, 1),
                        pipe.remove_keys('filename', 'path', 'width', 'height')
                        --pipe.patch_sampler(10, 10)
                        )
end



--[[
Coil.dataset = {
   data       = data_tensor
   object_ids = ids_tensor
   angles     = angle_tensor
}

data[i]
object_ids[i]
angles[i]


d = Dataset(Coil.load_dataset)
]]
