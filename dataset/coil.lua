require 'torch'
require 'image'
require 'paths'
require 'fs'
require 'nn'

require 'util'
require 'util/file'
require 'dataset/pipeline'
require 'fn'
require 'fn/seq'

require 'dataset'
require 'dataset/table_dataset'

--local Coil = torch.class("dataset.Coil")
Coil = {}

Coil.name         = 'coil'
Coil.dimensions   = {1, 128, 128}
Coil.n_dimensions = 3 * 128 * 128
Coil.size         = 7200
Coil.url          = 'http://somewhere.com/coil-100.t7.zip'
Coil.file         = 'coil-100.t7'

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


function processed_coil_images(dir, width, height)
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


function Coil.dataset()
   local pipeline = processed_coil_images('ext/coil-100', 64, 64)
   local table = pipe.to_data_table(100, pipeline)

   return dataset.TableDataset(table)
end
