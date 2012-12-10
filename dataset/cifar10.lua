require 'torch'
require 'nn'
require 'image'
require 'paths'

require 'util/file'
require 'logroll'
require 'dataset'

Cifar = {}
Cifar.name         = 'cifar10',
Cifar.dimensions   = {3, 32, 32},
Cifar.n_dimensions = 3 * 32 * 32,
Cifar.size         = function() return 50000 end,
Cifar.classes      = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'},
Cifar.url          = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz',
Cifar.dir          = 'cifar-10-batches-bin'
Cifar.files        = {'cifar-10-batches-bin/data_batch_1.bin', 'cifar-10-batches-bin/data_batch_2.bin',
                      'cifar-10-batches-bin/data_batch_3.bin', 'cifar-10-batches-bin/data_batch_4.bin',
                      'cifar-10-batches-bin/data_batch_5.bin'},
Cifar.batch_size   = 10000
Cifar.test_file = {'cifar-10-batches-bin/test_batch.bin'}

local function cifar_source(dir, pattern)
    local data = torch.Tensor(unpack(Cifar.dimensions))
    local files = pipe.matching_paths(dir, pattern)
    local f = torch.DiskFile('data_batch_1.bin', 'r')
    local class = f:readByte()

t7> img:storage():copy(f:readByte(3072))

end

function Cifar.data_source()
    local path = dataset.data_path(Cifar.name, Cifar.url, Cifar.dir)
    return cifar_source(path, "
end

function Cifar.test_source()
end

local function load_data_files(md)
    local data   = torch.Tensor(md.size(), md.n_dimensions)
    local labels = torch.Tensor(md.size())

    for i, file in ipairs(md.files) do
        local path = dataset.data_path(md.name, md.url, file)

        local subset = torch.load(path, 'ascii')
        data[  {{(i - 1) * md.batch_size + 1, i * md.batch_size}}] = subset.data:t():double()
        labels[{{(i - 1) * md.batch_size + 1, i * md.batch_size}}] = subset.labels
    end
    labels = labels + 1

    return data, labels
end


local function local_normalization(data)
    normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))

    for i = 1, cifar10_md.size() do
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


local function present_dataset(dataset)
    local labelvector = torch.zeros(10)

    util.set_index_fn(dataset,
      function(self, index)
          local input = dataset.data[index]
          local label = dataset.labels[index]
          local target = labelvector:zero()
          target[label] = 1

          local display = function()
              image.display{image=input, zoom=4, legend='cifar10[' .. index .. ']'}
          end

          return {input=input, target=target, label=label, display=display}
      end)

      util.set_size_fn(dataset,
        function(self)
            return self.size()
        end)

    return dataset
end

local function prepare_dataset(md)
    local data, labels = load_data_files(md)

    data = data:reshape(md.size(), 3, 32, 32)
    data = local_normalization(data)
    data = global_normalization(data)

    local dataset = util.merge(util.copy(md), {
        data   = data,
        labels = labels,
    })

    return present_dataset(dataset)
end


function cifar10.dataset()
    return prepare_dataset(cifar10_md)
end


function cifar10.raw_dataset()
    local data, labels = load_data_files(cifar10_md)

    data = data:reshape(cifar10_md.size(), 3, 32, 32)

    local dataset = util.merge(util.copy(cifar10_md), {
        data   = data,
        labels = labels,
    })

    return present_dataset(dataset)
end


function cifar10.test_dataset()
    return prepare_dataset(cifar10_test_md)
end


function cifar10.raw_test_dataset()
    local data, labels = load_data_files(cifar10_test_md)
    data = data:reshape(cifar10_test_md.size(), 3, 32, 32)

    local dataset = util.merge(util.copy(cifar10_test_md), {
        data   = data,
        labels = labels,
    })

    return present_dataset(dataset)
end


