require 'dataset/pipeline'
require 'paths'


local arg = require 'util/arg'

ImageSet = {}
function ImageSet.dataset(opts)

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
        return pipe.filteredpipeline(pipe.image_dir_source(dir,randomize),
                                    thres,
                                    unpack(p))
    else
        return pipe.pipeline(pipe.image_dir_source(dir,randomize),
                             unpack(p))
    end
end

