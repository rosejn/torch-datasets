require 'fs'
require 'paths'
require 'image'
require 'pprint'

require 'util'
require 'fn'
require 'fn/seq'
require 'dataset/table_dataset'

-- A dataset pipeline system, allowing for easy loading and transforming of
-- datasets.  A pipeline processes individual samples, which are just tables of
-- values (numbers, tensors, strings).  The only mandatory field is data, which
-- must be a torch.Tensor, and images should be in the form of
-- <channels x width x height>.
--
-- e.g.
--
--    { data     = <torch.Tensor>,
--      class    = class_id,
--      path     = 'obj2__45.png',
--      frame    = 9,
--      rotation = 45,
--      zoom     = 0.2
--    }

pipe = {}

--------------------------------------------------------------------------
-- Pipeline construction
--------------------------------------------------------------------------

-- Create a processing pipeline out of a table of
-- pipeline functions that should all take a sample and return a sample.
-- e.g.
--
--   local processor =
--    pipe.line({pipe.image_loader,
--              pipe.scaler(width, height),
--              pipe.rgb2yuv,
--              pipe.normalizer,
--              pipe.spatial_normalizer(1, 7, 1, 1),
--              patch_sampler(10, 10)
--              })
--
--   local data_stream = pipe.run(pipe.file_data_source(path), processor)
function pipe.line(funcs)
   return function(sample)
      return fn.thread(sample, unpack(funcs))
   end
end


-- Connect a sample source to a pipeline, returning a sample iterator.
function pipe.connect(src, line)
   return seq.map(line, src)
end


-- Create a processing pipeline that takes a data source iterator, and a list of
-- pipeline functions that should all take a sample and return a sample.
-- Returns a sample iterator.
--
-- e.g.
--
--    pipe.pipeline(pipe.file_data_source(path),
--                  pipe.image_loader,
--                  pipe.scaler(width, height),
--                  pipe.rgb2yuv,
--                  pipe.normalizer,
--                  pipe.spatial_normalizer(1, 7, 1, 1),
--                  patch_sampler(10, 10)
--                  )
function pipe.pipeline(src, ...)
   local funcs = {...}
   return pipe.connect(src, pipe.line(funcs))
end


-- Animate input samples by sending them through an animation pipeline
-- iteratively for n_frames.  (So each input sample will result in an
-- animation that is n_frames samples long.)
function pipe.animator(src, anim_line, n_frames)
   local framer = function(start)
      local frame = 1
      local cur_sample = start
      return function()
         s = cur_sample
         s.frame = frame
         frame = frame + 1
         local animated = anim_line(s)
         cur_sample = util.deep_copy(animated)
         return animated
      end
   end

   return seq.mapcat(function(sample)
                      print("sample.class: ", sample.class)
                      return seq.take(n_frames, framer(sample))
                    end,
                    src)
end


--------------------------------------------------------------------------
-- Sources
--------------------------------------------------------------------------


-- Returns a sequence of {filename = <fname>, path = <path> } entries for a
-- directory dir.  If a pattern p is also passed, then only files which match
-- the pattern will be returned.
-- (Note, it doesn't have to be a whole match, so even a suffix match will work.)
-- e.g.
--   pipe.matching_paths('./data', '.png')
--      => { { filename = 'image_1.png', path = 'data/image_1.png' } ... }
function pipe.file_source(dir, p)
  local files = seq.map(function(filename)
     return {
        filename = filename,
        path = paths.concat(dir, filename)
     }
  end,
  seq.seq(fs.readdir(dir)))

  if p then
     return seq.filter(function(s)
        return string.find(s.filename, p)
     end,
     files)
  else
     return files
  end
end


-- Returns a sequence of tables representing images in a directory, where
-- each table has the path and filename properties.  (Use the image_loader stage
-- to read the paths and load the images.)
function pipe.image_dir_source(dir)
   local files = pipe.file_source(dir)

   return seq.filter(function(s)
      local suffix = string.match(s.filename, '%.(%a+)$')
      return image.is_supported(suffix)
   end,
   files)
end


-- Turn a data table into a pipeline source, producing samples.
function pipe.data_table_source(table)
   local dataset = dataset.TableDataset(table)
   return dataset:sampler({shuffled = false})
end


-- Read data from a file on disk, returns a metadata table and a pipeline source.
function pipe.disk_object_source(path)
   local f = torch.DiskFile(path, 'r')
   f:binary()

   local size = f:readInt()
   local metadata = f:readObject()
   local count = 0

   local src = function()
      if count == size then
         return nil
      else
         local sample = f:readObject()
         count = count + 1
         return sample
      end
   end

   metadata.size = size
   return src, metadata
end



--------------------------------------------------------------------------
-- Pipeline Utilities
--------------------------------------------------------------------------

-- Load a data table from a file on disk.
function pipe.data_table_from_file(path)
   local md, src = pipe.disk_object_source(path)
   return pipe.data_table_sink(md.size, src)
end


-- Play a movie by displaying samples out of src at frame rate fps.
function pipe.animation_delay(fps)
   return function(sample)
      util.sleep(1.0 / fps)
      return sample
   end
end


-- Display a sample.
function pipe.display(options)
   local win
   options = options or {zoom = 5}

   return function(sample)
      if sample == nil then return nil end

      options.image = sample.data
      options.win = win
      win = image.display(options)
      return sample
   end
end


-- Pretty print a sample.
function pipe.pprint(sample)
   if sample == nil then return nil end

   pprint(sample)
   return sample
end


--------------------------------------------------------------------------
-- Pipeline Sinks
--------------------------------------------------------------------------

-- Copy samples in a pipeline into a data table which will be one table
-- with a tensor or table for each property of the samples.
-- (e.g. 100 samples each with {data = <tensor> class = id} will result
-- in one table with two tensors of size 100.)
-- The dtable argument is optional, and can be passed if you want to write
-- into an existing datatable, rather than allocating a new one.
function pipe.data_table_sink(n, pipeline, dtable)
   local sample = pipeline()

   -- Inspect the first sample to determine tensor types and dimensions,
   -- or reuse old dtable if available.
   if dtable == nil then
      dtable = {}

      for k,v in pairs(sample) do
         local store

         if type(v) == 'number' then
            store = torch.Tensor(n)
         elseif util.is_tensor(v) then
            store = torch.Tensor(unpack(util.concat({n}, v:size():totable())))
         else
            store = {}
         end
         dtable[k] = store
      end
   end

   for i, sample in seq.take(n, seq.indexed(seq.concat({sample}, pipeline))) do
      for k,v in pairs(sample) do
         if type(v) == 'number' or util.is_tensor(v) then
            dtable[k][i] = v
         else
            table.insert(dtable[k], v)
         end
      end
   end

   return dtable
end


-- Write a pipeline to disk.
function pipe.disk_object_sink(path, metadata)
   metadata = metadata or {}
   local file = torch.DiskFile(path, 'w')
   file:binary()

   file:writeInt(0)
   file:writeObject(metadata)

   local count = 0

   return function(sample)
      if sample == nil then
         print('end of sequence, writing size and closing file')
         file:seek(1)
         file:writeInt(count)
         file:close()
      else
         file:writeObject(util.deep_copy(sample))
         -- Once the force patch is accepted, we can switch to this:
         --file:writeObject(sample, true)
         count = count + 1
      end

      return sample
   end
end


--------------------------------------------------------------------------
-- Transformation stages
--------------------------------------------------------------------------


-- Convert from continuous to binary data.
function pipe.binarize(sample)
   sample.data = sample:gt(0.5):float()
end


-- Pad sample.data on all sides with value v.
function pipe.pad_values(width, v)
   return function(sample)
      local new_sample = torch.Tensor(sample:size(1),
                                      sample:size(2) + width*2,
                                      sample:size(3) + width*2):fill(v)
      new_sample:narrow(2, width+1, sample:size(2)):narrow(3, width+1, sample:size(3)):copy(sample)
      sample.data = new_sample
      return sample
   end
end


-- Reads sample.path and loads an image (torch.Tensor) into sample.data.
-- RGB images will will be (3 x WIDTH x HEIGHT) dimensions.
-- TODO: might need additional options here for specifying channels etc...
function pipe.image_loader(sample)
   if sample == nil then return nil end

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
   if sample == nil then return nil end

   sample.data = image.rgb2yuv(sample.data)
   return sample
end


-- Flip sample.data vertically
function pipe.flip_vertical(sample)
   if sample == nil then return nil end

   image.vflip(sample.data, sample.data:clone())
   return sample
end


-- Flip sample.data horizontally
function pipe.flip_horizontal(sample)
   if sample == nil then return nil end

   image.hflip(sample.data, sample.data:clone())
   return sample
end


-- Scales the sample.data to be the target width and height.
function pipe.scaler(width, height)
   return function(sample)
      if sample == nil then return nil end

      sample.data   = image.scale(sample.data, width, height)
      sample.width  = width
      sample.height = height
      return sample
   end
end


-- Crop sample.data to the rect defined by (x1, y1), (x2, y2)
function pipe.cropper(x1, y1, x2, y2)
   return function(sample)
      if sample == nil then return nil end

      sample.data = image.crop(sample.data, x1, y1, x2, y2)
      return sample
   end
end


-- Crop sample.data to a random patch of size width x height.
function pipe.patch_sampler(width, height)
   return function(sample)
      if sample == nil then return nil end

      local x = math.random(1, sample.width - width)
      local y = math.random(1, sample.height - height)
      sample.data   = image.crop(sample.data, x, y, x + width, y + height)
      sample.width  = width
      sample.height = height
      return sample
   end
end


-- Subtracts the mean and divides by the std for sample.data.
function pipe.normalizer(sample)
   local mean = sample.data:mean()
   local std  = sample.data:std()
   sample.data:add(-mean)
   sample.data:mul(1.0 / std)
   return sample
end


-- Using a gaussian kernel, locally subtract the mean and divide by standard
-- deviation.  (Highlight edges and remove areas of low frequency...)
-- e.g.
--   normalizer = pipe.spatial_normalizer(1, 7)
--   sample = normalizer(sample)
function pipe.spatial_normalizer(channel, radius, threshold, thresval)
   local neighborhood  = image.gaussian1D(radius)
   local normalizer    = nn.SpatialContrastiveNormalization(channel, neighborhood, threshold, thresval):float()

   return function(sample)
      if sample == nil then return nil end

      sample.data[{{channel},{},{}}]:copy(normalizer:forward(sample.data[{{channel},{},{}}]:float()))
      return sample
   end
end


-- Select a subset of keys from a sample table, discarding the rest.
function pipe.select_keys(...)
   local keys = {...}

   return function(sample)
      if sample == nil then return nil end

      local new_sample = {}
      for _, key in ipairs(keys) do
         new_sample[key] = sample[key]
      end

      return new_sample
   end
end


-- Remove one or more keys from a sample table.
function pipe.remove_keys(...)
   local keys = {...}

   return function(sample)
      if sample == nil then return nil end

      for _, key in ipairs(keys) do
         sample[key] = nil
      end

      return sample
   end
end


-- Typecast the value of property key in samples to type t.
-- (e.g. pipe.type('float', 'data') will typecast all data
-- tensors to be float tensors.)
function pipe.type(t, key)
   return function(sample)
      if key then
         sample[key] = sample[key][t](sample[key])
      else
         for k,_ in pairs(sample) do
            sample[k] = sample[key][t](sample[key])
         end
      end
      return sample
   end
end


-- Divides sample.data values by a constant factor
function pipe.div(n)
   local factor = 1.0 / n
   return function(sample)
      sample.data:mul(factor)
      return sample
   end
end


-- Filter a sequence based on some field
function pipe.filter(source, field, value)
   return seq.filter(
      function(sample)
         return sample[field] == value
      end,
      source
   )
end


-- Rotate input samples by r radians.
function pipe.rotator(r)
   --local rotated = torch.Tensor()
   return function(sample)
      --rotated:resizeAs(sample.data)
      local res = image.rotate(sample.data, r)
      sample.data:copy(res)
      return sample
   end
end


-- Translate input samples by dx, dy.
function pipe.translator(dx, dy)
   local translated = torch.Tensor()
   return function(sample)
      translated:resizeAs(sample.data)
      image.translate(translated, sample.data, dx, dy)
      sample.data:copy(translated)
      return sample
   end
end


-- Zoom input samples in or out by factor dz.
function pipe.zoomer(dz)
   local zoomed = torch.Tensor()

   return function(sample)
      local src_width  = sample.data:size(2)
      local src_height = sample.data:size(3)
      local width      = math.floor(src_width * dz)
      local height     = math.floor(src_height * dz)
      zoomed:resize(sample.data:size(1), width, height)

      image.scale(sample.data, zoomed, 'bilinear')
      if dz > 1 then
         local sx = math.floor((width - src_width) / 2)+1
         local sy = math.floor((height - src_height) / 2)+1
         sample.data:copy(res:narrow(2, sx, src_width):narrow(3, sy, src_height))
      else
         local sx = math.floor((src_width - width) / 2)+1
         local sy = math.floor((src_height - height) / 2)+1
         sample.data:zero()
         sample.data:narrow(2, sx, width):narrow(3, sy, height):copy(zoomed)
      end

      return sample
   end
end


--[[
-- TODO: Integrate some of the random animation capability that was available in
-- this mnist animation code, so we can produce randomized movies.

function Mnist:_animate(rotation, translation, zoom)
   local full_size = self.frames * self.size
   local animated = torch.Tensor(full_size, Mnist.n_dimensions):zero()
   local animated_labels = torch.Tensor(full_size)

   for i=1,self.size do
      for f=1,self.frames do
         animated_labels[1 + (i-1)*self.frames + (f-1)] = self.labels[i]
      end
   end

   for sample=1,self.size do
      local transformers = {}
      if (#rotation > 0) then
         local rot_start, rot_finish = dataset.rand_pair(rotation[1], rotation[2])
         rot_start = rot_start * math.pi / 180
         rot_finish = rot_finish * math.pi / 180
         local rot_delta = (rot_finish - rot_start) / self.frames
         table.insert(transformers, dataset.rotator(rot_start, rot_delta))
         self.rotation = rotation
      end

      if (#translation > 0) then
         local xmin_tx, xmax_tx = dataset.rand_pair(translation[1], translation[2])
         local ymin_tx, ymax_tx = dataset.rand_pair(translation[3], translation[4])
         local dx = (xmax_tx - xmin_tx) / frames
         local dy = (ymax_tx - ymin_tx) / frames
         table.insert(transformers, dataset.translator(xmin_tx, ymin_tx, dx, dy))
         self.translation = translation
      end

      if (#zoom > 0) then
         local zoom_start, zoom_finish = dataset.rand_pair(zoom[1], zoom[2])
         local zoom_delta = (zoom_finish - zoom_start) / self.frames
         table.insert(transformers, dataset.zoomer(zoom_start, zoom_delta))
         self.zoom = zoom
      end

      local src = self.samples[sample]:unfold(1, 28, 28)
      for f=1,self.frames do
         local dst = animated:narrow(1, (sample-1)*self.frames + f, 1):select(1,1):unfold(1, 28, 28)
         local tsrc = src
         for _, transform in ipairs(transformers) do
            transform(tsrc, dst)
            tsrc = dst
         end
      end
   end
end

--]]
