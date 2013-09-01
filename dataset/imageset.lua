require 'dataset/pipeline'
require 'dataset/SimpleDataset'


local arg = require 'util/arg'

ImageSet = {}
function ImageSet.pipe(opts)

    opts = opts or {}
    local dir          = arg.required(opts,'dir','string')
    local patch_width  = arg.optional(opts,'width',0)
    local patch_height = arg.optional(opts,'height',patch_width)
    local do_lcn       = arg.optional(opts,'lcn',true)
    local do_yuv       = arg.optional(opts,'yuv',true)
    local do_gray      = arg.optional(opts,'gray',not do_yuv)
    local lcn_chn      = arg.optional(opts,'lcn_channel',0)
    local std_thres    = arg.optional(opts,'std_thres',0.2)
    local label_file   = arg.optional(opts,'label',nil)
    local randomize    = arg.optional(opts,'randomize',true)
    local loop         = arg.optional(opts,'loop',true)


    if not dir or not paths.dirp(dir) then
        error('directory does not exist : ' .. dir)
    end
    if do_yuv and do_gray then
        error('I can not do YUV and Grayscale conversion at the same time')
    end
    if lcn_chn == 0 then
        lcn_chn = nil
    end

    local p = {}
    table.insert(p,pipe.image_loader)
    if do_yuv then
        table.insert(p,pipe.rgb2yuv)
    end
    if do_gray then
        table.insert(p,pipe.rgb2gray)
    end
    if do_lcn then
        table.insert(p,pipe.lcn())
    end
    if patch_height > 0 and patch_width > 0 then
        table.insert(p,pipe.patch_sampler(patch_width,patch_height))
    end

    if label_file and not paths.filep(label_file) then
        error('label file not found : ' .. label_file)
    end

    table.insert(p,pipe.gc())

    if std_thres > 0 then
        local thres = function(sample)
            if torch.std(sample.data) > std_thres then
                return true
            else
                return false
            end
        end
        return pipe.filteredpipeline(pipe.image_dir_source(dir,randomize,loop),
                                    thres,
                                    unpack(p))
    else
        return pipe.pipeline(pipe.image_dir_source(dir,randomize,loop),
                             unpack(p))
    end
end

function ImageSet.dataset(opts)
    opts = opts or {}
    local dir          = arg.required(opts,'dir','string')
    local label_file   = arg.optional(opts,'label',nil)
    local name         = arg.optional(opts,'name','Image Dataset : ' .. dir)
    local do_lcn       = arg.optional(opts,'lcn',true)
    local do_yuv       = arg.optional(opts,'yuv',true)
    local do_gray      = arg.optional(opts,'gray',not do_yuv)
    local lcn_chn      = arg.optional(opts,'lcn_channel',nil)
    local patch_width  = arg.optional(opts,'width',0)
    local patch_height = arg.optional(opts,'height',patch_width)
    local std_thres    = arg.optional(opts,'std_thres',0)

    local data = {}
    local path = {}
    local class

    if do_yuv and do_gray then
        error('I can not do YUV and Grayscale conversion at the same time')
    end
    if lcn_chn == 0 then
        lcn_chn = nil
    end
    local p = {}
    table.insert(p,pipe.image_loader)
    if do_yuv then
        table.insert(p,pipe.rgb2yuv)
    end
    if do_gray then
        table.insert(p,pipe.rgb2gray)
    end
    if do_lcn then
        table.insert(p,pipe.lcn())
    end

    if label_file and not paths.filep(label_file) then
        error('label file not found : ' .. label_file)
    end

    -- load images into memory
    local datapipe = pipe.pipeline(pipe.image_dir_source(dir), unpack(p))
    for sample in datapipe do
        table.insert(data,sample.data)
        table.insert(path,sample.path)
        io.write(string.format('\r %d %s',#data, sample.path))
        io.flush()
    end
    io.write('\n')
    print('Loaded ' .. #data .. ' samples')

    -- load labels if they exist
    if label_file then
        class = {}
        for line in io.lines(label_file) do
            local sample,label = line:match('^(.+)%s+(.+)$')
            if not sample or not label then
                error('Formatting error in label file : ' .. label)
            end
            class[sample] = label
        end
        print('Loaded labels')
    end

    local data_table = {data = data, path=path, classes = class}
    local meta_data = {name = name}

    local dataset = dataset.SimpleDataset(data_table,meta_data)

    local postpipe = nil
    if patch_width > 0 and patch_height > 0 then
        postpipe = pipe.line({pipe.patch_sampler(patch_width,patch_height)})
    end
    local sampler = dataset:sampler({shuffled = true, pipeline = postpipe})

    local thres = function(sample)
        if std_thres == 0 or torch.std(sample.data) > std_thres then
            return true
        else
            return false
        end
    end
    local fsampler = seq.filter(thres,sampler)

    return dataset, sampler
end

