#include "neuralNetwork.h"
//COMENTARIO DE PRUEBA
int main(){
    srand((unsigned int)time(NULL));
    const double learningRate = 0.1f;

    double hiddenLayer[numHiddenNodes];
    double outputLayer[numOutputs];

    double hiddenLayerBias[numHiddenNodes];
    double outputLayerBias[numOutputs];
    // basicamente esto es lo que son las flechas que une a cada nodo
    double hiddenWeights[numInputs][numHiddenNodes];
    double outputWeights[numHiddenNodes][numOutputs];

    // Añadimos el data set
    double trainingInputs[numTrainingSets][numInputs] = {
            {0.0, 0.0},
            {0.0, 1.0},
            {1.0, 0.0},
            {1.0, 1.0}
    };

    double trainingOutputs[numTrainingSets][numOutputs] = {
            {0.0},
            {1.0},
            {1.0},
            {0.0}
    };

    initNN(hiddenWeights, outputWeights, hiddenLayerBias, outputLayerBias);

    trainNetwork(hiddenLayer, outputLayer,trainingInputs, trainingOutputs, hiddenWeights, outputWeights,
                 hiddenLayerBias, outputLayerBias, learningRate);


    printf("Final Hidden Weights\n[ ");
    for (int j = 0; j < numHiddenNodes; j++)
    {
        printf("[ ");
        for (int k = 0; k < numInputs; k++)
        {
            printf("%f ", hiddenWeights[k][j]);
        }
        printf("] ");
    }
    printf("]\nFinal Hidden Biases\n[ ");
    for (int j = 0; j < numHiddenNodes; j++)
    {
        printf("%f ", hiddenLayerBias[j]);
    }
    printf("]\nFinal Output Weights\n");
    for (int j = 0; j < numOutputs; j++)
    {
        printf("[ ");
        for (int k = 0; k < numHiddenNodes; k++)
        {
            printf("%f ", outputWeights[k][j]);
        }
        printf("]\n");
    }

    printf("Final Output Biases\n[ ");
    for (int j = 0; j < numOutputs; j++)
    {
        printf("%f ", outputLayerBias[j]);
    }
    printf("]\n");

    printf("\n--- Final predictions ---\n");
    for (int i = 0; i < numTrainingSets; i++) {
        forwardPass(trainingInputs[i], hiddenWeights, outputWeights,
                    hiddenLayerBias, outputLayerBias,
                    hiddenLayer, outputLayer);

        printf("Input: %.1f %.1f  ->  Output: %.4f\n",
               trainingInputs[i][0], trainingInputs[i][1], outputLayer[0]);
    }




    return 0;
}

double initWeights(){
    return (double)rand() / (double)RAND_MAX;
}

double sigmoid(double x){
    return 1.0 / (1.0 + exp(-x));
}

double dsigmoid(double x){
    return x * (1.0 - x);
}

void initNN(double hiddenWeights[numInputs][numHiddenNodes],
            double outputWeights[numHiddenNodes][numOutputs],
            double hiddenLayerBias[numHiddenNodes],
            double outputLayerBias[numOutputs]){
    int i, j;

    for(i = 0; i < numInputs; i++){
        for(j = 0; j < numHiddenNodes; j++){
            hiddenWeights[i][j] = initWeights();
        }
    }

    for(i = 0; i < numHiddenNodes; i++){
        for(j = 0; j < numOutputs; j++){
            outputWeights[i][j] = initWeights();
        }
    }

    for(i = 0; i < numHiddenNodes; i++){
        hiddenLayerBias[i] = initWeights();
    }

    for(i = 0; i < numOutputs; i++){
        outputLayerBias[i] = initWeights();
    }
}

void swap(int *a, int *b){
    int temp;
    temp = *a;
    *a = *b;
    *b = temp;
}

void shuffle(int *array, size_t n){
    if(n > 1){
        for(size_t i = 0; i < n; i++){
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            swap(&array[i], &array[j]);
        }
    }
}
// Esta parte es la que "piensa" toma las entradas, la combina con los pesos y bias, las pasa por la sigmoide y genera la prediccion
void forwardPass(
        const double trainingInputs[numInputs],
        double hiddenWeights[numInputs][numHiddenNodes],
        double outputWeights[numHiddenNodes][numOutputs],
        const double hiddenLayerBias[numHiddenNodes],
        const double outputLayerBias[numOutputs],
        double hiddenLayer[numHiddenNodes],
        double outputLayer[numOutputs]
){
    //Hidden Layer activation
    for(int j = 0; j < numHiddenNodes; j++){
        double activation = hiddenLayerBias[j];
        for(int k = 0; k < numInputs; k++){
            activation += trainingInputs[k] * hiddenWeights[k][j];
        }
        hiddenLayer[j] = sigmoid(activation);
    }
    //Output Layer activation
    for(int j = 0; j < numOutputs; j++){
        double activation = outputLayerBias[j];
        for(int k = 0; k < numHiddenNodes; k++){
            activation += hiddenLayer[k] * outputWeights[k][j];
        }
        outputLayer[j] = sigmoid(activation);
    }
}
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
){
    double deltaOutput[numOutputs];
    double deltaHidden[numHiddenNodes];
    double error;
    // Calcular la delta de la capa de salida
    for(int j = 0; j < numOutputs; j++){
        error = trainingOutputs[j] - outputLayer[j];
        deltaOutput[j] = error * dsigmoid(outputLayer[j]);
    }
    // Calcular la delta de la capa oculta
    for(int j = 0; j < numHiddenNodes; j++){
        error = 0.0f;
        for(int k = 0; k < numOutputs; k++){
            error += deltaOutput[k] * outputWeights[j][k];
        }
        deltaHidden[j] = error * dsigmoid(hiddenLayer[j]);
    }

    // Actualizar los pesos y sesgo de la capa de salida
    for(int j = 0; j < numOutputs; j++){
        outputLayerBias[j] += deltaOutput[j] * learningRate;
        for(int k = 0; k < numHiddenNodes; k++){
            outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * learningRate;
        }
    }
    // Actualizar pesos y sesgo de la capa oculta
    for(int j = 0; j < numHiddenNodes; j++){
        hiddenLayerBias[j] += deltaHidden[j] * learningRate;
        for(int k = 0; k < numInputs; k++){
            hiddenWeights[k][j] += trainingInputs[k] * deltaHidden[j] * learningRate;
        }
    }
}

void trainNetwork(
        double hiddenLayer[numHiddenNodes],
        double outputLayer[numOutputs],
        double trainingInputs[numTrainingSets][numInputs],
        double trainingOutputs[numTrainingSets][numOutputs],
        double hiddenWeights[numInputs][numHiddenNodes],
        double outputWeights[numHiddenNodes][numOutputs],
        double hiddenLayerBias[numHiddenNodes],
        double outputLayerBias[numOutputs],
        double learningRate){

    int trainingSetOrder[numTrainingSets] = {0,1,2,3};

    for(int epoch = 0; epoch < numEpochs; epoch++){
        shuffle(trainingSetOrder, numTrainingSets);
        for(int x = 0; x < numTrainingSets; x++){
            int i = trainingSetOrder[x];
            // Esta parte es la que "piensa" toma las entradas, la combina con los pesos y bias,
            // las pasa por la sigmoide y genera la prediccion.
            forwardPass(trainingInputs[i], hiddenWeights, outputWeights, hiddenLayerBias, outputLayerBias, hiddenLayer, outputLayer);
            // Esta parte mide cuando se equivoco la red y cuando contribuyo cada neurona a ese error
            // para luego poder ajustar los pesos en la direccion correcta y aprender.
            backwardPass(trainingInputs[i], trainingOutputs[i], hiddenLayer, outputLayer,hiddenWeights, outputWeights,hiddenLayerBias, outputLayerBias,learningRate);
            // Mostrar progreso cada 1000 épocas
        }
        if (epoch % 1000 == 0) {
            printf("\nEpoch %d\n", epoch);
            for (int i = 0; i < numTrainingSets; i++) {
                forwardPass(trainingInputs[i], hiddenWeights, outputWeights,
                            hiddenLayerBias, outputLayerBias, hiddenLayer, outputLayer);
                printf("Input: %.1f %.1f -> Output: %.4f\n",
                       trainingInputs[i][0], trainingInputs[i][1], outputLayer[0]);
            }
            printf("---------------------------\n");
        }
    }
}