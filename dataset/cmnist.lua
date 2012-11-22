require 'torch'
require 'image'
require 'paths'
require 'dok'

require 'fn/seq'
require 'util'
require 'util/file'

require 'dataset'
require 'debugger'

local Mnist = torch.class("dataset.Mnist")
Mnist.name         = 'mnist'
Mnist.dimensions   = {1, 28, 28}
Mnist.n_dimensions = 1 * 28 * 28
Mnist.size         = 60000
Mnist.classes      = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
Mnist.url          = 'http://data.neuflow.org/data/mnist-th7.tgz'
Mnist.file         = 'mnist-th7/train.th7'

Mnist.test_file    = 'mnist-th7/test.th7'
Mnist.test_size    = 10000

-- Get the raw, unprocessed dataset.
-- Returns a 60,000 x 785 tensor, where each image is 28*28 = 784 values in the
-- range [0-255], and the 785th element is the class ID.
function Mnist.raw_data(n)
   local path = dataset.data_path(Mnist.name, Mnist.url, Mnist.file)
   local n_examples, n_dimensions, data = dataset.load_data_file(path, n)
   return data
end

-- Get the raw, unprocessed test dataset.
-- Returns a 10,000 x 785 tensor in the same format as the training set
-- described above.
function Mnist.raw_test_data(n)
   local path = dataset.data_path(Mnist.name, Mnist.url, Mnist.test_file)
   local n_examples, n_dimensions, data = dataset.load_data_file(path, n)
   return data
end

local function rand_between(min, max)
   return math.random() * (max - min) + min
end

local function rand_pair(v_min, v_max)
   local a = rand_between(v_min, v_max)
   local b = rand_between(v_min, v_max)
   --local start = math.min(a, b)
   --local finish = math.max(a, b)
   --return start, finish
   return a,b
end

local function rotator(start, delta)
   local angle = start
   return function(src, dst)
      image.rotate(dst, src, angle)
      angle = angle + delta
   end
end

local function translator(startx, starty, dx, dy)
   local started = false
   local cx = startx
   local cy = starty
   return function(src, dst)
      local res = image.translate(src, cx, cy)
      dst:copy(res)
      cx = cx + dx
      cy = cy + dy
   end
end

local function zoomer(start, dz)
   local factor = start
   return function(src, dst)
      local src_width  = src:size()[1]
      local src_height = src:size()[2]
      local width      = math.floor(src_width * factor)
      local height     = math.floor(src_height * factor)

      local res = image.scale(src, width, height)
      if factor > 1 then
         local sx = math.floor((width - src_width) / 2)+1
         local sy = math.floor((height - src_height) / 2)+1
         dst:copy(res:sub(sx, sx+src_width-1, sy, sy+src_height-1))
      else
         local sx = math.floor((src_width - width) / 2)+1
         local sy = math.floor((src_height - height) / 2)+1
         dst:zero()
         dst:sub(sx,  sx+width-1, sy, sy+height-1):copy(res)
      end

      factor = factor + dz
   end
end

function Mnist:__init(opts)
   local animated
   local scale, normalize, size, frames, rotation, translation, zoom
   --[[
   local _, scale, normalize, size, frames, rotation, translation, zoom = dok.unpack({...},
         'Mnist:__init',
         'returns a pre-processed MNIST dataset',
         {arg='scale',       type='table',   help='scale dataset within a range {min, max}', default={}},
         {arg='normalize',   type='boolean', help='apply global normalization => (data - mean) / std', default=false},
         {arg='size',        type='number',  help='specify a size if you only want a subset of the data', default=Mnist.size},
         {arg='frames',      type='number',  help='specify the number of frames to animate each sample', default=10},
         {arg='rotation',    type='table',   help='rotation parameters = {min, max}', default=nil},
         {arg='translation', type='table',   help='translation parameters = {xmin, xmax, ymin, ymax}', default=nil},
         {arg='zoom',        type='table',   help='scaling parameters = {min, max}', default=nil})
         ]]
   scale = opts.scale or {}
   normalize = opts.normalize or false
   size = opts.size or Mnist.size
   frames = opts.frames or 10
   rotation = opts.rotation
   translation = opts.translation
   zoom = opts.zoom

   print("scale: ",       scale)
   print("rotation: ",    rotation)
   print("translation: ", translation)
   print("zoom: ",        zoom)

   local data = Mnist.raw_data(size)
   local sample_data = data:narrow(2, 1, Mnist.n_dimensions)
   local class_ids = torch.Tensor(size)
   for i=1,size do
      class_ids[i] = data[{i, 785}]
   end

   if normalize then
       mean, std = dataset.global_normalization(sample_data)
   end

   if (#scale > 0) then
      dataset.scale(sample_data, scale[1], scale[2])
   end

   if rotation or translation or zoom then
      animated = torch.Tensor(frames * size, Mnist.n_dimensions):zero()

      for sample=1,size do
         local transformers = {}
         if rotation then
            local rot_start, rot_finish = rand_pair(rotation[1], rotation[2])
            rot_start = rot_start * math.pi / 180
            rot_finish = rot_finish * math.pi / 180
            local rot_delta = (rot_finish - rot_start) / frames
            table.insert(transformers, rotator(rot_start, rot_delta))
         end

         if translation then
            local xmin_tx, xmax_tx = rand_pair(translation[1], translation[2])
            local ymin_tx, ymax_tx = rand_pair(translation[3], translation[4])
            local dx = (xmax_tx - xmin_tx) / frames
            local dy = (ymax_tx - ymin_tx) / frames
            table.insert(transformers, translator(xmin_tx, ymin_tx, dx, dy))
         end

         if zoom then
            local zoom_start, zoom_finish = rand_pair(zoom[1], zoom[2])
            local zoom_delta = (zoom_finish - zoom_start) / frames
            table.insert(transformers, zoomer(zoom_start, zoom_delta))
         end

         local src = sample_data[sample]:unfold(1, 28, 28)
         for f=1,frames do
            local dst = animated:narrow(1, (sample-1)*frames + f, 1):select(1,1):unfold(1, 28, 28)
            local tsrc = src
            for _, transform in ipairs(transformers) do
               transform(tsrc, dst)
               tsrc = dst
            end
         end
      end
   end

   self.size      = size
   self.frames    = frames
   self.class_ids = class_ids
   self.data      = animated
end

-- Returns a sequence of animations, where each animation is a tensor of size
-- frames * dimensions.
function Mnist:animations()
   return seq.map(function(i) return self.data:narrow(1, (i-1)*self.frames+1, self.frames) end, seq.range(self.size))
end
