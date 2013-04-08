require 'torch'
require 'image'
require 'paths'
require 'dok'

require 'fn'
require 'fn/seq'
require 'util'
require 'util/file'
require 'util/arg'
local arg = util.arg

require 'dataset'
require 'dataset/TableDataset'
require 'dataset/whitening'


Mnist = {}

Mnist.name         = 'mnist'
Mnist.dimensions   = {1, 28, 28}
Mnist.n_dimensions = 1 * 28 * 28
Mnist.size         = 60000
Mnist.classes      = {[0] = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
Mnist.url          = 'http://data.neuflow.org/data/mnist-th7.tgz'
Mnist.file         = 'mnist-th7/train.th7'

Mnist.test_file    = 'mnist-th7/test.th7'
Mnist.test_size    = 10000

local function load_data_file(path, n)
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

-- Get the raw, unprocessed dataset.
-- Returns a 60,000 x 785 tensor, where each image is 28*28 = 784 values in the
-- range [0-255], and the 785th element is the class ID.
function Mnist.raw_data(n)
   local path = dataset.data_path(Mnist.name, Mnist.url, Mnist.file)
   local n_examples, n_dimensions, data = load_data_file(path, n)
   return data
end


-- Get the raw, unprocessed test dataset.
-- Returns a 10,000 x 785 tensor in the same format as the training set
-- described above.
function Mnist.raw_test_data(n)
   local path = dataset.data_path(Mnist.name, Mnist.url, Mnist.test_file)
   local n_examples, n_dimensions, data = load_data_file(path, n)
   return data
end


-- Setup an MNIST dataset instance.
--
--   m = dataset.Mnist()
--
--   -- scale values between [0,1] (by default they are in the range [0,255])
--   m = dataset.Mnist({scale = {0, 1}})
--
--   -- or normalize (subtract mean and divide by std)
--   m = dataset.Mnist({normalize = true})
--
--   -- only import a subset of the data (imports full 60,000 samples otherwise)
--   m = dataset.Mnist({size = 1000})
--
--   -- use the test data rather than the training data:
--   m = dataset.Mnist({test = true})
function Mnist.dataset(opts)
   local scale, normalize, zca_whiten, size, frames, rotation, translation, zoom
   opts          = opts or {}
   test          = arg.optional(opts, 'test', false)
   scale         = arg.optional(opts, 'scale', {})
   normalize     = arg.optional(opts, 'normalize', false)
   binarize     = arg.optional(opts, 'binarize', false)
   zca_whiten    = arg.optional(opts, 'zca_whiten', false)
   size          = arg.optional(opts, 'size', test and Mnist.test_size or Mnist.size)
   sort          = arg.optional(opts, 'sort', false)
   transform     = arg.optional(opts, 'sort', nil)

   local transformations = {}

   local data
   if test then
      data = Mnist.raw_test_data(size)
   else
      data = Mnist.raw_data(size)
   end

   local samples = data:narrow(2, 1, Mnist.n_dimensions):clone()
   samples:resize(size, unpack(Mnist.dimensions))

   local labels = torch.Tensor(size)
   for i=1,size do
      labels[i] = data[{i, 785}]
   end

   if sort then
      samples, labels = dataset.sort_by_class(samples, labels)
   end

   if (#scale == 2) then
      dataset.scale(samples, scale[1], scale[2])
   end

   local d = dataset.TableDataset({data = samples, class = labels}, Mnist)

   if binarize then
      if #scale == 2 then
          threshold = (scale[2]+scale[1])/2
      else 
          threshold = 128
      end
      d:binarize(threshold)
   end

   if normalize then
      d:normalize_globally()
   end

   if zca_whiten then
      d:zca_whiten()
   end

   return d

end
