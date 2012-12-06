require 'fn/seq'
require 'dataset/pipeline'
require 'dataset/coil'
require 'util'
require 'pprint'


function test_coil_from_images()
   pipe.movie_player(seq.take(200, coil_images('ext/coil-100', 64, 64)), 10)
end


function test_coil_mini_batches()
   local table
   local image_data = coil_images('ext/coil-100', 64, 64)

   for i=1,2 do
      table = pipe.to_data_table(100, image_data, table)

      print('data table:')
      pprint(table)
      print('class: ', table.class[1])
   end
end


--test_coil_from_images()
test_coil_mini_batches()
