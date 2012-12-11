require 'paths'
require 'torch'
require 'image'

require 'util'
require 'util/file'

local TORCH_DIR = paths.concat(os.getenv('HOME'), '.torch')
local DATA_DIR  = paths.concat(TORCH_DIR, 'data')

dataset = {}

-- Check locally and download dataset if not found.  Returns the path to the
-- downloaded data file.
function dataset.get_data(name, url)
  local set_dir   = paths.concat(DATA_DIR, name)
  local data_file = paths.basename(url)
  local data_path = paths.concat(set_dir, data_file)

  --print("checking for file located at: ", data_path)

  check_and_mkdir(TORCH_DIR)
  check_and_mkdir(DATA_DIR)
  check_and_mkdir(set_dir)
  check_and_download_file(data_path, url)

  return data_path
end


-- Downloads the data if not available locally, and returns local path.
function dataset.data_path(name, url, file)
    local data_path  = dataset.get_data(name, url)
    local data_dir   = paths.dirname(data_path)
    local local_path = paths.concat(data_dir, file)

    if not is_file(local_path) then
        do_with_cwd(data_dir,
          function()
              decompress_file(data_path)
          end)
    end

    return local_path
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

