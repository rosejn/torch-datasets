require 'dataset/pipeline'
require 'dataset/SimpleDataset'
require 'ffmpeg'

local arg = require 'util/arg'

VideoSet = {}
function VideoSet.pipe(opts)

    opts = opts or {}
    local dir          = arg.required(opts,'dir','string')
    local patch_width  = arg.optional(opts,'width',0)
    local patch_height = arg.optional(opts,'height',patch_width)
    local do_lcn       = arg.optional(opts,'lcn',true)
    local do_yuv       = arg.optional(opts,'yuv',true)
    local do_gray      = arg.optional(opts,'gray',not do_yuv)
    local lcn_chn      = arg.optional(opts,'lcn_channel',0)
    local std_thres    = arg.optional(opts,'std_thres',0.0)
    local label_file   = arg.optional(opts,'label',nil)
    local suffix       = arg.optional(opts,'suffix','avi')
    local randomize    = arg.optional(opts,'randomize',false)
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
        return pipe.filteredpipeline(pipe.video_dir_source(dir,suffix,randomize,loop),
                                    thres,
                                    unpack(p))
    else
        return pipe.pipeline(pipe.video_dir_source(dir,suffix,randomize,loop),
                             unpack(p))
    end
end

function VideoSet.dataset(opts)
    opts = opts or {}
    local dir          = arg.required(opts,'dir','string')
    local suffix       = arg.optional(opts,'suffix','avi')
    local label_file   = arg.optional(opts,'label',nil)
    local name         = arg.optional(opts,'name','Video Dataset : ' .. dir)
    local do_lcn       = arg.optional(opts,'lcn',true)
    local do_yuv       = arg.optional(opts,'yuv',true)
    local do_gray      = arg.optional(opts,'gray',not do_yuv)
    local lcn_chn      = arg.optional(opts,'lcn_channel',nil)
    local patch_width  = arg.optional(opts,'width',0)
    local patch_height = arg.optional(opts,'height',patch_width)
    local patch_x1     = arg.optional(opts,'x1',0)
    local patch_y1     = arg.optional(opts,'y1',0)
    local patch_x2     = arg.optional(opts,'x2',0)
    local patch_y2     = arg.optional(opts,'y2',0)
    local size_width   = arg.optional(opts,'resize_width',320)
    local size_height  = arg.optional(opts,'resize_height',240)

    local std_thres    = arg.optional(opts,'std_thres',0)

    local nframes      = arg.optional(opts,'nframes',0)
    local fps          = arg.optional(opts,'fps',10)

    local maxframes = math.max(nframes,12000)
    local video_length = maxframes/fps

    local data = {}
    local path = {}
    local class

    if do_yuv and do_gray then
        error('I can not do YUV and Grayscale conversion at the same time')
    end
    if lcn_chn == 0 then
        lcn_chn = nil
    end
    if label_file and not paths.filep(label_file) then
        error('label file not found : ' .. label_file)
    end

    -- load images into memory
    local datapipe = pipe.file_source(dir,'%.'..suffix..'$')
    for sample in datapipe do
        -- table.insert(data,sample.data)
        table.insert(path,sample.path)
        io.write(string.format('\r %d %s          ',#path, sample.path))
        io.flush()
    end
    io.write('\n')
    print('Processing ' .. #path .. ' files')

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

    local data_table = {data = path, path=path, classes = class}
    local meta_data = {name = name}

    local dataset = dataset.SimpleDataset(data_table,meta_data)


    -- we need to override the sample function to be able loop over
    -- same file for a certain number of steps before jumping to next one.

    local frame_counter = 0
    local file_counter = 0
    local sample = {}
    function dataset:sample(i)

        if file_counter == 0 or nframes > 0 and frame_counter == nframes or nframes == 0 and frame_counter >= sample.ffmpeg.nframes then
            file_counter = math.floor(torch.uniform(1,self:size()))
            for key, v in pairs(self.dataset) do
                sample[key] = v[file_counter]
            end
            local oprint = print
            print = function () end
            sample.ffmpeg = ffmpeg.Video{path=sample.path, fps=fps, length=video_length, width=size_width, height=size_height}
            print = oprint
            if patch_width > 0 and patch_height > 0 then
                patch_x1 = patch_x2
                patch_y1 = patch_y2
            end
            frame_counter = 0
            if nframes > 0 and nframes < sample.ffmpeg.nframes then
                sample.ffmpeg.current = math.random(0,sample.ffmpeg.nframes-nframes-1)
            end
        elseif nframes > 0 and frame_counter == sample.ffmpeg.nframes then
            if patch_width > 0 and patch_height > 0 then
                patch_x1 = patch_x2
                patch_y1 = patch_y2
            end
        end

        frame_counter = frame_counter + 1
        sample.frame = frame_counter
        sample.data = sample.ffmpeg:forward()
        return sample
    end


    local p = {}
    if do_yuv then
        table.insert(p,pipe.rgb2yuv)
    end
    if do_gray then
        table.insert(p,pipe.rgb2gray)
    end
    if do_lcn then
        table.insert(p,pipe.lcn())
    end

    if patch_x1 > 0 and patch_y1 > 0 and patch_x2 > patch_x1 and patch_y2 > patch_y1 then
        table.insert(p,pipe.cropper(patch_x1, patch_y1, patch_x2, patch_y2))
    elseif patch_width > 0 and patch_height > 0 then
        local cropper = function(sample)
            if sample == nil then return nil end
            local width = sample.data:size(3)
            local height = sample.data:size(2)
            if patch_x1 == patch_x2 and patch_y1 == patch_y2 then
                patch_x1 = math.random(1,width-patch_width)
                patch_y1 = math.random(1,height-patch_height)
                patch_x2 = patch_x1 + patch_width -1
                patch_y2 = patch_y1 + patch_height -1
            end
            sample.data  = image.crop(sample.data, patch_x1, patch_y1, patch_x2, patch_y2)
            return sample
        end
        table.insert(p,cropper)
    end

    local postpipe = nil
    if #p > 0 then
        postpipe = pipe.line(p)
    end
    sampler = dataset:sampler({pipeline = postpipe})

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


