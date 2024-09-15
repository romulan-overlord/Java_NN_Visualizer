public class NeuralNetwork {
    private Layer[] layers;

    public NeuralNetwork(int... layerSizes) {
        layers = new Layer[layerSizes.length - 1];
        for (int i = 0; i < layers.length; i++) {
            layers[i] = new Layer(layerSizes[i + 1], layerSizes[i]);
        }
    }

    // Feedforward through all layers
    public double[] feedForward(double[] inputs) {
        for (Layer layer : layers) {
            inputs = layer.feedForward(inputs);
        }
        return inputs;
    }

    // Backpropagation
    public void backpropagate(double[] inputs, double[] expectedOutputs, double learningRate) {
        // Feedforward to calculate outputs
        double[] outputs = feedForward(inputs);

        // Calculate output layer error (delta)
        for (int i = 0; i < layers[layers.length - 1].neurons.length; i++) {
            Neuron neuron = layers[layers.length - 1].neurons[i];
            neuron.delta = (expectedOutputs[i] - outputs[i]) *
                    ActivationFunction.sigmoidDerivative(neuron.output);
        }

        // Calculate hidden layer errors (delta)
        for (int i = layers.length - 2; i >= 0; i--) {
            Layer layer = layers[i];
            Layer nextLayer = layers[i + 1];

            for (int j = 0; j < layer.neurons.length; j++) {
                Neuron neuron = layer.neurons[j];
                double sum = 0.0;
                for (int k = 0; k < nextLayer.neurons.length; k++) {
                    sum += nextLayer.neurons[k].weights[j] * nextLayer.neurons[k].delta;
                }
                neuron.delta = sum * ActivationFunction.sigmoidDerivative(neuron.output);
            }
        }

        // Update weights and biases
        for (int i = 0; i < layers.length; i++) {
            Layer layer = layers[i];
            double[] input = (i == 0) ? inputs : layers[i - 1].feedForward(inputs);

            for (Neuron neuron : layer.neurons) {
                for (int j = 0; j < neuron.weights.length; j++) {
                    neuron.weights[j] += learningRate * neuron.delta * input[j];
                }
                neuron.bias += learningRate * neuron.delta;
            }
        }
    }
}
