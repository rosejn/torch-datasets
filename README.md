# Datasets

A collection of easy to use datasets for training and testing machine learning
algorithms with Torch7.


## Usage

    require('dataset/mnist')
    d = mnist.dataset()
    d.size()                      -- => 60000
    d[100]                        -- => {[input]  = DoubleTensor - size: 784
                                         [target] = DoubleTensor - size: 10
                                         [label]  = 1}

    td = mnist.test_dataset()
    td.size()                     -- => 10000

Each dataset should provide a dataset() function that returns a table.  This
table consists of:

* size:       function returning size of dataset (mandatory)
* __index:    the index operator (mydataset[i]) which returns data items (mandatory)

Each data item is a table consisting of:

* input:     tensor data value (mandatory)
* target:    target tensor value
* label:     data label (unspecified type)
