require 'torch'
require 'image'
require 'paths'
require 'fs'

require 'util/file'
require 'dataset'
require 'fn/seq'

coil = {}

coil_md = {
    name         = 'coil',
    dimensions   = {1, 128, 128},
    n_dimensions = 1 * 128 * 128,
    size         = function() return 7200 end,

    classes      = {1},

    url          = 'http://somewhere.com/coil-100-t7.tgz',
    file         = 'coil-100.t7'
}

function coil.as_tensor(src_dir)
   local files = fs.readdir(src_dir)

   local obj_files = seq.filter(function(name) return string.find(name, "obj") end, files)
   local library = torch.Tensor(7200, 3, 128, 128)

   for fname in obj_files do
      _, _, img, angle = string.find(fname, "obj(%d+)__(%d+).ppm")
      img = tonumber(img)
      angle = tonumber(angle)
      local index = img + (angle / 5)

      local data = image.load(paths.concat(src_dir, fname))
      library[{index, {}, {}, {}}]:copy(data)
   end

   return library
end

-- Convert the original coil-100 dataset from a directory of images
-- to a serialized torch tensor.
function coil.save_thz(src_dir, file)
   local data = coil.as_tensor(src_dir)
   torch.save(file, data)
end
