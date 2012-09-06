require 'nn'
require 'paths'

require 'util'
require 'dataset/util'

house_numbers = {
  name    = 'house_numbers',
  classes = {'1','2','3','4','5','6','7','8','9','0'},
  url     = 'http://data.neuflow.org/data/housenumbers/train_32x32.t7',
  size    = function()
    return 10000
  end,

  test_url  = 'http://data.neuflow.org/data/housenumbers/test_32x32.t7',
  test_size = function()
    return 2000
  end
}

house_numbers.filename = paths.basename(house_numbers.url)

-- Load the data from disk, downloading if necessary, and in this case
-- transpose to column major.
function house_numbers.rgb_data(raw_data)
  local data_path = dataset.get_data(house_numbers.name, house_numbers.url)
  local raw_data  = torch.load(data_path, 'ascii')
  local labels    = raw_data.y[1]
  local data      = raw_data.X:transpose(3,4)
  data            = data:float()

  return labels, data
end

-- Convert to YUV colorspace to easily access the brightness (Y) channel.
function house_numbers.yuv_data()
  local labels, data = house_numbers.rgb_data()
  local yuv = dataset.rgb_to_yuv(data)

  return labels, yuv
end

-- Returns the mean and standard deviation of n_channels of pixel data
function house_numbers.channel_stats(data, n_channels)
  local mean = {}
  local std  = {}

  for i=1, n_channels do
    mean[i] = data[{ {},i,{},{} }]:mean()
    std[i]  = data[{ {},i,{},{} }]:std()
  end

  return mean, std, data
end

-- Normalizes the data in place and returns: mean, std.
-- First per channel, global normalization is shifted by subtracting the mean and
-- scaled by dividing by the standard deviation.
-- Local normaliztion the Y channel by locally using a neighborhood gaussian
function house_numbers.normalize_data(data, n_channels)
  local mean, std = house_numbers.channel_stats(data, n_channels)

  for i=1, n_channels do
    data[{ {},i,{},{} }]:add(-mean[i])
    data[{ {},i,{},{} }]:div(std[i])
  end

  local neighborhood  = image.gaussian1D(7)
  local normalization = nn.SpatialContrastiveNormalization(1, neighborhood):float()

  for i = 1, n_channels do
    data[{ i,{1},{},{} }] = normalization:forward(data[{ i,{1},{},{} }])
  end

  return mean, std
end

-- Returns a normalized, YUV dataset.
local function prepare_dataset(path)
  local channels     = {'y','u','v'}
  local labels, data = house_numbers.yuv_data()
  local mean, std    = house_numbers.normalize_data(data, #channels)

  local dataset = concat(house_numbers, {
    data     = data,
    channels = channels,
    mean     = mean,
    std      = std,
  })

  local labelvector = torch.zeros(10)

  util.set_index_fn(dataset,
    function(self, i)
        local label = labels[i]
        local target = labelvector:zero()
        target[label + 1] = 1
        return {input=data[i], target=target, label=labels[i]}
    end)

  return dataset
end

function house_numbers.dataset()
end

-- TODO: finish me
function house_numbers.test_dataset()

end

function house_numbers.display(dataset, n_samples)
  local n_samples = n_samples or 32

  local ys = dataset.data[{ {1, n_samples},1 }]
  local us = dataset.data[{ {1, n_samples},2 }]
  local vs = dataset.data[{ {1, n_samples},3 }]
  local n_rows = math.ceil(math.sqrt(n_samples))

  image.display{image=ys, nrow=n_rows, zoom=2, legend='house_numbers: Y'}
  image.display{image=us, nrow=n_rows, zoom=2, legend='house_numbers: U'}
  image.display{image=vs, nrow=n_rows, zoom=2, legend='house_numbers: V'}
end

