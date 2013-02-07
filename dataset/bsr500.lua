require 'torch'
require 'image'
require 'paths'
require 'fs'
require 'nn'

require 'util'
require 'util/file'
require 'dataset/pipeline'
require 'fn'
require 'fn/seq'

require 'dataset'
require 'dataset/TableDataset'


BSR = {}

BSR.name         = 'BSR'
BSR.dimensions   = {3, 128, 128}
BSR.n_dimensions = 3 * 128 * 128
BSR.size         = 7200
BSR.url          = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
BSR.file         = 'BSR'

BSR.default_options = {
    size      = BSR.size,
    resize    = {64, 64},
    yuv       = true,
    normalize = true,
    whitening = {1, 7, 1, 1},
}

function bsr_image_dir()
	return paths.concat(dataset.data_path(BSR.name, BSR.url, BSR.file), "BSDS500/data/images/train/")
end

function bsr_image_patches(patch_width, patch_height)
    local dir = bsr_image_dir()
    return pipe.pipeline(pipe.image_dir_source(dir),
                         pipe.image_loader,
                         pipe.patch_sampler(patch_width, patch_height))
end

