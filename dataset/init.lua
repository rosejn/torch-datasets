require 'paths'
require 'torch'
require 'image'

require 'util'
require 'util/file'

local DMT_DIR = paths.concat(os.getenv('HOME'),'.dmt')

dataset = {
  data_dir = paths.concat(DMT_DIR, 'data')
}


-- Check locally and download dataset if not found.  Returns the path to the
-- downloaded data file.
function dataset.get_data(name, url)
  local set_dir   = paths.concat(dataset.data_dir, name)
  local data_file = paths.basename(url)
  local data_path = paths.concat(set_dir, data_file)

  check_and_mkdir(DMT_DIR)
  check_and_mkdir(dataset.data_dir)
  check_and_mkdir(set_dir)
  check_and_download_file(data_path, url)

  return data_path
end

-- Downloads the data if not available locally, and returns local path.
function dataset.data_path(name, url, file, decompress)
    local data_path  = dataset.get_data(name, url)
    local data_dir   = paths.dirname(data_path)
    local local_path = paths.concat(data_dir, file)
    decompress = decompress or decompress_tarball

    if not is_file(local_path) then
        do_with_cwd(data_dir, function() decompress(data_path) end)
    end

    return local_path
end


function dataset.load_data_file(path, n)
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


-- Convert pixel data in place (destructive) from RGB to YUV colorspace.
function dataset.rgb_to_yuv(pixel_data)
    for i = 1, pixel_data:size()[1] do
        pixel_data[i] = image.rgb2yuv(pixel_data[i])
    end

    return pixel_data
end

function dataset.stats(data)
    local mean = data:mean()
    local std  = data:std()

    return mean, std
end

function dataset.global_normalization(data)
    local mean, std = dataset.stats(data)

    data:add(-mean)
    data:mul(1/std)

    return mean, std
end

function dataset.scale(data, min, max)
    local range = max - min
    local dmin = data:min()
    local dmax = data:max()
    local drange = dmax - dmin

    data:add(-dmin)
    data:mul(range)
    data:mul(1/drange)
    data:add(min)
end

function dataset.contrastive_normalization(plane, data)
    local normalize = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
end

function dataset.print_stats(dataset)
    for i,channel in ipairs(dataset.channels) do
        mean = dataset.data[{ {},i }]:mean()
        std  = dataset.data[{ {},i }]:std()

        print('data, '..channel..'-channel, mean: ' .. mean)
        print('data, '..channel..'-channel, standard deviation: ' .. std)
    end
end

function dataset.rand_between(min, max)
   return math.random() * (max - min) + min
end

function dataset.rand_pair(v_min, v_max)
   local a = dataset.rand_between(v_min, v_max)
   local b = dataset.rand_between(v_min, v_max)
   --local start = math.min(a, b)
   --local finish = math.max(a, b)
   --return start, finish
   return a,b
end

function dataset.sort_by_class(samples, labels)
    local size = labels:size()[1]
    local sorted_labels, sort_indices = torch.sort(labels)
    local sorted_samples = torch.Tensor(samples:size())

    for i=1, size do
        sorted_samples[i] = samples[sort_indices[i]]
    end

    return sorted_samples, sorted_labels
end

