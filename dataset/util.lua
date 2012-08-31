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

-- Convert pixel data in place (destructive) from RGB to YUV colorspace.
function dataset.rgb_to_yuv(pixels, size)
    for i = 1, size do
        pixels[i] = image.rgb2yuv(pixels[i])
    end

    return pixels
end

function dataset.stats(data)
    local mean = data:mean()
    local std  = data:std()

    return mean, std
end


function dataset.print_stats(dataset)
    for i,channel in ipairs(dataset.channels) do
        mean = dataset.data[{ {},i }]:mean()
        std  = dataset.data[{ {},i }]:std()

        print('data, '..channel..'-channel, mean: ' .. mean)
        print('data, '..channel..'-channel, standard deviation: ' .. std)
    end
end

