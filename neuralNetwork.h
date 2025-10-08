#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


#define numInputs 2
#define numOutputs 1
#define numHiddenNodes 2
#define numTrainingSets 4
#define numEpochs 10000

double initWeights();
double sigmoid(double x);
double dsigmoid(double x);
void initNN(double hiddenWeights[numInputs][numHiddenNodes],
            double outputWeights[numHiddenNodes][numOutputs],
            double hiddenLayerBias[numHiddenNodes],
            double outputLayerBias[numOutputs]);
void swap( int *a, int *b);
void shuffle(int *array, size_t n);
void forwardPass(
        const double trainingInputs[numInputs],
        double hiddenWeights[numInputs][numHiddenNodes],
        double outputWeights[numHiddenNodes][numOutputs],
        const double hiddenLayerBias[numHiddenNodes],
        const double outputLayerBias[numOutputs],
        double hiddenLayer[numHiddenNodes],
        double outputLayer[numOutputs]
);
void backwardPass(
        const double trainingInputs[numInputs],
        const double trainingOutputs[numOutputs],
        double hiddenLayer[numHiddenNodes],
        double outputLayer[numOutputs],
        double hiddenWeights[numInputs][numHiddenNodes],
        double outputWeights[numHiddenNodes][numOutputs],
        double hiddenLayerBias[numHiddenNodes],
        double outputLayerBias[numOutputs],
        double learningRate
);
void trainNetwork(
        double hiddenLayer[numHiddenNodes],
        double outputLayer[numOutputs],
        double trainingInputs[numTrainingSets][numInputs],
        double trainingOutputs[numTrainingSets][numOutputs],
        double hiddenWeights[numInputs][numHiddenNodes],
        double outputWeights[numHiddenNodes][numOutputs],
        double hiddenLayerBias[numHiddenNodes],
        double outputLayerBias[numOutputs],
        double learningRate);

#endif