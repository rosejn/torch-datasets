# Datasets

A collection of easy to use datasets for training and testing machine learning
algorithms with Torch7.


## Usage

    require('dataset/mnist')
    m = dataset.Mnist()
    d.size                        -- => 60000
    d:sample(100)                 -- => tensor, label

    -- scale values between [0,1] (by default they are in the range [0,255])
    m = dataset.Mnist({scale = {0, 1}})

    -- or normalize (subtract mean and divide by std)
    m = dataset.Mnist({normalize = true})

    -- only import a subset of the data (imports full 60,000 samples otherwise)
    m = dataset.Mnist({size = 1000})

    -- optionally animate mnist digits using translation, rotation, and
    -- scaling over a certain number of frames.
    m = dataset.Mnist{frames = frames,
                      rotation = {-20, 20},
                      zoom = {0.3, 1.5},
                      translation = {-8, 8, -8, 8}
                     }

To process a randomly shuffled ordering of the dataset:

    for sample, label in m:samples() do
      net:forward(sample)
    end

Or access mini batches:

    local batch, labels = m:mini_batch(1)

    -- or use directly
    net:forward(m:mini_batch(1))

    -- set the batch size using an options table
    local batch, labels = m:mini_batch(1, {size = 100})


To process the full dataset in randomly shuffled mini-batches:

    for batch, labels in m:mini_batches() do
       net:forward(batch)
    end


If the dataset was created with animated transformations, these animation
sequences can be accessed individually or in shuffled order as well:

    for frame,label in m:animation(1) do
       local img = frame:unfold(1,28,28)
       win = image.display({win=win, image=img, zoom=10})
       util.sleep(1 / 24)
    end


    for anim in m:animations() do
       for frame,label in anim do
          local img = frame:unfold(1,28,28)
          win = image.display({win=win, image=img, zoom=10})
          util.sleep(1 / 24)
       end
    end
