require 'fn/seq'
require 'dataset/pipeline'
require 'dataset/coil'
require 'util'

function test_coil_from_images()
   pipe.movie_player(seq.take(200, coil_images('ext/coil-100', 64, 64)), 10)
end

test_coil_from_images()
