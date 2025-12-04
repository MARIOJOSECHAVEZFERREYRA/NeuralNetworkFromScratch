cmake_minimum_required(VERSION 4.0)
project(NeuralNetworkFromScratch C)

set(CMAKE_C_STANDARD 11)

include_directories(.)

add_executable(NeuralNetworkFromScratch
        neuralNetwork.c
        neuralNetwork.h)
