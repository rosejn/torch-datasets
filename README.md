# Datasets

A collection of easy to use datasets for training and testing machine learning
algorithms with Torch7.


## Usage

    require('dataset/mnist')
    m = Mnist.dataset()
    d:size()                      -- => 60000
    d:sample(100)                 -- => {data = tensor, class = label}

    -- scale values between [0,1] (by default they are in the range [0,255])
    m = dataset.Mnist({scale = {0, 1}})

    -- or normalize (subtract mean and divide by std)
    m = dataset.Mnist({normalize = true})

    -- only import a subset of the data (imports full 60,000 samples otherwise),
    -- sorted by class label
    m = dataset.Mnist({size = 1000, sort = true})


To process a randomly shuffled ordering of the dataset:

    for sample in m:sampler() do
      net:forward(sample.data)
    end


Or access mini batches:

    local batch = m:mini_batch(1)

    -- or use directly
    net:forward(m:mini_batch(1).data)

    -- set the batch size using an options table
    local batch = m:mini_batch(1, {size = 100})


To process the full dataset in randomly shuffled mini-batches:

    for batch in m:mini_batches() do
       net:forward(batch.data)
    end


Generate animations over 10 frames for each sample, which will
randomly rotate, translate, and/or zoom within the ranges passed.

    local anim_options = {
        frames      = 10,
        rotation    = {-20, 20},
        translation = {-5, 5, -5, 5},
        zoom        = {0.6, 1.4}
     }
     s = dataset:sampler({animate = anim_options})


Standard pipeline options can be used to add post-processing stages:

     s = dataset:sampler({pad = 5, binarize = true, type = 'float'})


Pass a custom pipeline for processing samples:

     s = dataset:sampler({pipeline = my_pipeline})
