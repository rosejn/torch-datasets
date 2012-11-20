require 'torch'
require 'image'
require 'paths'
require 'fs'

require 'util'
require 'util/arg'
require 'util/file'
require 'dataset'
require 'fn/seq'

local arg = util.arg
local Coil = torch.class("dataset.Coil")

Coil.name         = 'coil',
Coil.dimensions   = {1, 128, 128},
Coil.n_dimensions = 3 * 128 * 128,
Coil.size         = function() return 7200 end,
Coil.url          = 'http://somewhere.com/coil-100.t7.zip',
Coil.file         = 'coil-100.t7'

function Coil:__init(options)
   self.md = coil_md
   local path = dataset.data_path(self.md.name, self.md.url, self.md.file)
   local n_objects, n_dimensions, data = dataset.load_data_file(path)
   self.data = data
end

function Coil:raw_rgb_data()
   return n_examples, n_dimensions, data
end

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


-- Read in the original COIL-100 dataset as ppm files and write out a torch
-- data file.
--
-- The file has a 2 element header:
--    int: n_images
--    int: n_dimensions
--
-- And then n_images entries of the form:
--    float: rgb_data (n_dimensions - 2 in size)
--    float: object_id (0-100)
--    float: angle     (0-360 in increments of 5)
function Coil.convert_to_torch(src_dir, path)
   local f = torch.DiskFile(path, 'w')
   f:binary()
   f:writeInt(coil_md.size())
   f:writeInt(coil_md.n_dimensions + 2)

   local files = fs.readdir(src_dir)
   local obj_files = seq.filter(function(name) return string.find(name, "obj") end, files)

   for fname in obj_files do
      _, _, img, angle = string.find(fname, "obj(%d+)__(%d+).ppm")
      img = tonumber(img)
      angle = tonumber(angle)
      local index = img + (angle / 5)

      local data = image.load(paths.concat(src_dir, fname))
      f:writeFloat(data:storage())
      f:writeFloat(img)
      f:writeFloat(angle)
   end

   f:close()
end

