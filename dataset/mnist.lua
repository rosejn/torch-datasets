require 'torch'
require 'image'
require 'paths'

require 'util/file'
require 'dataset'

mnist = {}

mnist_md = {
    name         = 'mnist',
    dimensions   = {1, 28, 28},
    n_dimensions = 1 * 28 * 28,
    size         = function() return 60000 end,

    classes      = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0},

    url          = 'http://data.neuflow.org/data/mnist-th7.tgz',
    file         = 'mnist-th7/train.th7'
}

mnist_test_md = util.merge(util.copy(mnist_md), {
    size         = function() return 10000 end,
    file         = 'mnist-th7/test.th7'
})


local function load_data_file(path)
    local f = torch.DiskFile(path, 'r')
    f:binary()

    local n_examples   = f:readInt()
    local n_dimensions = f:readInt()
    local tensor       = torch.Tensor(n_examples, n_dimensions)
    tensor:storage():copy(f:readFloat(n_examples * n_dimensions))

    return n_examples, n_dimensions, tensor
end


-- Downloads the data if not available locally, and returns local path.
local function prepare_dataset(md, options)
    local path = dataset.data_path(md.name, md.url, md.file)
    local n_examples, n_dimensions, data = load_data_file(path)
    
    if options.convolutional then
        convo_sample = torch.Tensor(unpack(md.dimensions))
    elseif options.linscale01 then
    	local m = torch.min(data)  
	local M = torch.max(data) 
	data = (data - m) / (M - m)	
	print(type(data))
    elseif options.original==nil then
    	local mean, std = dataset.global_normalization(data:narrow(2, 1, n_dimensions - 1))
    end

    local labelvector = torch.zeros(10)

    local convo_sample

    local dataset = util.merge(util.copy(mnist_md), {
        data     = data,
        channels = {'y'},
        mean     = mean,
        std      = std,
        size     = function() return n_examples end,
        n_dimensions = n_dimensions - 1,
    })


    util.set_index_fn(dataset,
      function(self, index)
          local input = data[index]:narrow(1, 1, n_dimensions - 1):double()
          local label = data[index][n_dimensions]
          local target = labelvector:zero()
          target[label + 1] = 1

          local display = function()
              image.display{image=input:unfold(unpack(dataset.dimensions)),
                            zoom=4, legend='mnist[' .. index .. ']'}
          end

          if options.convolutional then
              convo_sample:copy(input)
              return {
                  input   = convo_sample,
                  target  = target,
                  label   = label,
                  display = display
              }
          else
              return {
                  input   = input,
                  target  = target,
                  label   = label,
                  display = display
              }
          end
      end)

      util.set_size_fn(dataset,
        function(self)
            return self.size()
        end)

    return dataset
end


function mnist.dataset(options)
    options = options or {}
    return prepare_dataset(mnist_md, options)
end


function mnist.test_dataset(options)
    options = options or {}
    return prepare_dataset(mnist_test_md, options)
end

