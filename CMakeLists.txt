CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
FIND_PACKAGE(Torch REQUIRED)

SET(src)
SET(luasrc dataset/init.lua dataset/mnist.lua dataset/cifar10.lua
    dataset/house_numbers.lua dataset/smallnorb.lua)

ADD_TORCH_PACKAGE(dataset "${src}" "${luasrc}" "Torch friendly datasets.")
