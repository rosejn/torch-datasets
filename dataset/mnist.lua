require 'torch'
require 'image'
require 'paths'

require 'util/file'
require 'logroll'
require 'dataset/util'

mnist = {
    name       = 'mnist',
    classes    = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0},
    url        = 'http://data.neuflow.org/data/mnist-th7.tgz',
    train_file = 'mnist-th7/train.th7',
    test_file  = 'mnist-th7/test.th7'
}

local function load_data_file(path)
    local f = torch.DiskFile(path, 'r')
    f:binary()

    local n_examples   = f:readInt()
    local n_dimensions = f:readInt()
    local tensor       = torch.Tensor(n_examples, n_dimensions)
    tensor:storage():copy(f:readFloat(n_examples * n_dimensions))

    return n_examples, n_dimensions, tensor
end

local function normalize_data(data)
    local mean, std = dataset.stats(data)

    data:add(-mean)
    data:mul(1/std)

    return mean, std
end

-- Downloads the data if not available locally, and returns local path.
local function data_path(file)
    local data_path  = dataset.get_data(mnist.name, mnist.url)
    local data_dir   = paths.dirname(data_path)
    local local_path = paths.concat(data_dir, file)

    if not is_file(local_path) then
        do_with_cwd(data_dir, function() decompress_tarball(data_path) end)
    end

    return local_path
end

local function prepare_dataset(path)
    local n_examples, n_dimensions, data = load_data_file(path)
    local mean, std = normalize_data(data:narrow(2, 1, n_dimensions - 1))
    local labelvector = torch.zeros(10)

    local dataset = util.concat(mnist, {
        data     = data,
        channels = {'y'},
        mean     = mean,
        std      = std,
        size     = function() return n_examples end,
        n_dimensions = n_dimensions - 1,
    })

    util.set_index_fn(dataset,
      function(self, index)
          local input = data[index]:narrow(1, 1, n_dimensions - 1)
          local label = data[index][n_dimensions]
          local target = labelvector:zero()
          target[label + 1] = 1
          return {input=input, target=target, label=label}
      end)

    return dataset
end

function mnist.dataset()
    local train_path = data_path(mnist.train_file)
    return prepare_dataset(train_path)
end

function mnist.test_dataset()
    local test_path = data_path(mnist.test_file)
    return prepare_dataset(test_path)
end

function mnist.display(dset, index)
  local index = index or 1
  local example = dset[index]
  image.display{image=example.input:unfold(1, 28, 28), zoom=2, legend='mnist'}
end

