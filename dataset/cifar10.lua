require 'torch'
require 'nn'
require 'image'
require 'paths'

require 'util/file'
require 'logroll'
require 'dataset'

require 'debugger'

cifar10 = {
    name         = 'cifar10',
    n_dimensions = 32 * 32 * 3,
    size         = function() return 50000 end,
    test_size    = function() return 10000 end,

    classes      = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'},

    url          = 'http://data.neuflow.org/data/cifar-10-torch.tar.gz',

    train_files  = {'cifar-10-batches-t7/data_batch_1.t7', 'cifar-10-batches-t7/data_batch_2.t7',
                    'cifar-10-batches-t7/data_batch_3.t7', 'cifar-10-batches-t7/data_batch_4.t7',
                    'cifar-10-batches-t7/data_batch_5.t7'},
    batch_size   = 10000,

    test_files   = {'cifar-10-batches-t7/test_batch.t7'}
}

local function load_data_files(files)
    local data   = torch.Tensor(cifar10.size(), cifar10.n_dimensions)
    local labels = torch.Tensor(cifar10.size())

    for i, file in ipairs(files) do
        local path = dataset.data_path(cifar10.name, cifar10.url, file)

        local subset = torch.load(path, 'ascii')
        data[  {{(i - 1) * cifar10.batch_size + 1, i * cifar10.batch_size}}] = subset.data:t():double()
        labels[{{(i - 1) * cifar10.batch_size + 1, i * cifar10.batch_size}}] = subset.labels
    end
    labels = labels + 1

    return data, labels
end


local function local_normalization(data)
    normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))

    for i = 1, cifar10.size() do
        -- rgb -> yuv
        local yuv = image.rgb2yuv(data[i])

        -- normalize y locally:
        yuv[1] = normalization(yuv[{{1}}])
        data[i] = yuv
    end

    return data
end


local function global_normalization(data)
    -- normalize u globally:
    mean_u = data[{ {},2,{},{} }]:mean()
    std_u = data[{ {},2,{},{} }]:std()
    data[{ {},2,{},{} }]:add(-mean_u)
    data[{ {},2,{},{} }]:div(-std_u)

    -- normalize v globally:
    mean_v = data[{ {},3,{},{} }]:mean()
    std_v = data[{ {},3,{},{} }]:std()
    data[{ {},3,{},{} }]:add(-mean_v)
    data[{ {},3,{},{} }]:div(-std_v)

    return data
end

local function prepare_dataset(files)
    local data, labels = load_data_files(files)

    data = data:reshape(cifar10.size(), 3, 32, 32)
    data = local_normalization(data)
    data = global_normalization(data)

    local dataset = util.concat(cifar10, {
        data = data,
        labels = labels,
    })

    return dataset
end

-- TODO: finish implementing these index functions copied from mnist

function cifar10.dataset()
    local dataset = prepare_dataset(cifar10.train_files)

    --[[
    util.set_index_fn(dataset,
      function(self, index)
          local input = data[index]:narrow(1, 1, n_dimensions - 1)
          local label = data[index][n_dimensions]
          local target = labelvector:zero()
          target[label + 1] = 1
          return {input=input, target=target, label=label}
      end)
      ]]

    return dataset
end


function cifar10.test_dataset()
    local dataset = prepare_dataset(cifar10.test_files)
    dataset.size = cifar10.test_size

    --[[
    util.set_index_fn(dataset,
      function(self, index)
          local input = data[index]:narrow(1, 1, n_dimensions - 1)
          local label = data[index][n_dimensions]
          local target = labelvector:zero()
          target[label + 1] = 1
          return {input=input, target=target, label=label}
      end)
      ]]

    return dataset
end

function cifar10.display(dset, index)
  local index = index or 1
  local example = dset[index]
  image.display{image=example.input:unfold(1, 28, 28), zoom=2, legend='mnist'}
end
