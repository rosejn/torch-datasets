require 'image'
require 'util'
require 'dataset/cmnist'

local win

local FPS = 24
local frames = 3 * 24

m = dataset.Mnist{size=10,
                  frames = frames,
                  rotation = {-20, 20},
                  zoom = {0.3, 1.5},
                  translation = nil -- {-8, 8, 8, 8}
                 }

for anim in m:animations() do
   for frame=1,frames do
      local img = anim[frame]:unfold(1,28,28)
      win = image.display({win=win, image=img, zoom=10})
      util.sleep(1 / 24)
   end
end

