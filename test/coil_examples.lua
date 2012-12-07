require 'fn/seq'
require 'dataset/pipeline'
require 'dataset/coil'
require 'util'
require 'pprint'


function coil_from_images()
   pipe.movie_player(seq.take(200, Coil.default_pipeline('ext/coil-100', 64, 64)), 30)
end


function coil_mini_batches()
   local table
   local image_data = Coil.default_pipeline('ext/coil-100', 64, 64)

   for i=1,2 do
      table = pipe.to_data_table(100, image_data, table)

      print('data table:')
      pprint(table)
      print('class: ', table.class[1])
   end
end


function coil_animations()
   local fps = 30
   local animation = pipe.line({
      --pipe.translator(2, 0),
      --pipe.zoomer(0.99),
      pipe.rotator(math.pi / 36)
   })
   local frames = pipe.animator(seq.take_nth(72, Coil.default_pipeline('ext/coil-100', 64, 64)), animation, fps)
   pipe.movie_player(seq.take(5*fps, frames), fps)
end

--coil_from_images()
--coil_mini_batches()
--coil_animations()


function coil_serialization()
   local p = Coil.default_pipeline('ext/coil-100', 64, 64)
   pipe.write_to_disk('test.dat', p, {foo = 'testing'})
   local dtable = pipe.data_table_from_file('test.dat')
   pprint(dtable)
end

coil_serialization()

