require 'torch'
require 'image'
require 'util'
require 'dataset/mnist'
require 'fn/seq'

local win

local FPS = 24
local frames = 2 * 24

function test_sampler()
   m = Mnist.dataset({size = 3})

   --a,b = m:mini_batch(2, 3)
   for sample, label in seq.take(12, m:sampler()) do
      print(type(sample), label)
   end
end


function test_animation()
   local fps = 30

   d = Mnist.dataset({size = 10})
   local anim_options = {
      frames        = fps,
      rotation    = {-20, 20},
      translation = {-5, 5, -5, 5},
      zoom        = {0.6, 1.4}
   }

   s = d:sampler({animate = anim_options})

   local win
   local i = 0
   for sample in seq.take(200, s) do
      i = i + 1
      win = image.display({win=win, image=sample.data, zoom=4})
      util.sleep(1 / fps)
   end
end


test_animation()
