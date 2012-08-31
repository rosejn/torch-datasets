CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
FIND_PACKAGE(Torch REQUIRED)

SET(src init.c)

SET(luasrc
    dataset/mnist.lua)

ADD_TORCH_PACKAGE(dataset "${luasrc}" "Datasets")
ADD_TORCH_DOK(dok dataset "3rd Party" "Prepared datasets, yes!" 111111.)
