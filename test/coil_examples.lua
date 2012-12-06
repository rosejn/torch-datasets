require 'fn/seq'
require 'dataset/pipeline'
require 'dataset/coil'
require 'util'
require 'pprint'


function test_coil_from_images()
   pipe.movie_player(seq.take(200, processed_coil_images('ext/coil-100', 64, 64)), 30)
end


function test_coil_mini_batches()
   local table
   local image_data = processed_coil_images('ext/coil-100', 64, 64)

   for i=1,2 do
      table = pipe.to_data_table(100, image_data, table)

      print('data table:')
      pprint(table)
      print('class: ', table.class[1])
   end
end

function coil_animations()
   local animation = pipe.line({pipe.rotator(math.pi / 36)})
   local frames = pipe.animator(processed_coil_images('ext/coil-100', 64, 64), animation, 30)
   pipe.movie_player(seq.take(200, frames))
end

test_coil_from_images()
--test_coil_mini_batches()
--coil_animations()
