import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;

public class NeuralNetwork {
    public List<Layer> layers;
    private double learningRate;

    public NeuralNetwork(int inputSize, int[] hiddenSizes, int outputSize, String[] activations, double learningRate) {
        this.learningRate = learningRate;
        layers = new ArrayList<>();

        // Create hidden and output layers
        int previousSize = inputSize;
        for (int i = 0; i < hiddenSizes.length; i++) {
            layers.add(new Layer(previousSize, hiddenSizes[i], activations[i]));
            previousSize = hiddenSizes[i];
        }
        layers.add(new Layer(previousSize, outputSize, "sigmoid")); // Use sigmoid in the output layer
    }

    // Forward pass
    public List<double[]> forward(double[] input) {
        List<double[]> activations = new ArrayList<>();
        activations.add(input);
        for (Layer layer : layers) {
            double[] z = new double[layer.outputSize];
            for (int i = 0; i < layer.outputSize; i++) {
                z[i] = 0;
                for (int j = 0; j < layer.inputSize; j++) {
                    z[i] += input[j] * layer.weights[i][j];
                }
                z[i] += layer.bias[i];
            }
            input = new double[layer.outputSize];
            for (int i = 0; i < layer.outputSize; i++) {
                input[i] = layer.activate(z[i]);
            }
            activations.add(input);
        }
        return activations;
    }

    // Backpropagation with binary cross-entropy loss
    public void backpropagate(double[] input, double[] target) {
        List<double[]> activations = forward(input);
        double[] outputActivation = activations.get(activations.size() - 1);

        double[][] deltas = new double[layers.size()][];

        // Compute delta for output layer using binary cross-entropy loss derivative
        deltas[layers.size() - 1] = new double[layers.get(layers.size() - 1).outputSize];
        for (int i = 0; i < layers.get(layers.size() - 1).outputSize; i++) {
            double y = target[i];
            double a = outputActivation[i];
            deltas[layers.size() - 1][i] = (a - y) / (a * (1 - a)); // BCE derivative for sigmoid output
        }

        // Backpropagate through the hidden layers
        for (int l = layers.size() - 2; l >= 0; l--) {
            deltas[l] = new double[layers.get(l).outputSize];
            for (int i = 0; i < layers.get(l).outputSize; i++) {
                double deltaSum = 0.0;
                for (int j = 0; j < layers.get(l + 1).outputSize; j++) {
                    deltaSum += deltas[l + 1][j] * layers.get(l + 1).weights[j][i];
                }
                deltas[l][i] = deltaSum * layers.get(l).activateDerivative(activations.get(l + 1)[i]);
            }
        }

        System.out.println("printing deltas: ");
        System.out.println(Arrays.deepToString(deltas));

        // Update weights and biases
        for (int l = 0; l < layers.size(); l++) {
            for (int i = 0; i < layers.get(l).outputSize; i++) {
                for (int j = 0; j < layers.get(l).inputSize; j++) {
                    layers.get(l).weights[i][j] -= learningRate * deltas[l][i] * activations.get(l)[j];
                }
                layers.get(l).bias[i] -= learningRate * deltas[l][i];
            }
        }
    }

    // Calculate Binary Cross-Entropy Loss
    public double calculateLoss(double[][] inputs, double[][] targets) {
        double totalLoss = 0.0;
        for (int i = 0; i < inputs.length; i++) {
            List<double[]> activations = forward(inputs[i]);
            double[] output = activations.get(activations.size() - 1);
            for (int j = 0; j < output.length; j++) {
                double y = targets[i][j];
                double a = output[j];
                totalLoss += -y * Math.log(a) - (1 - y) * Math.log(1 - a); // Binary cross-entropy loss
            }
        }
        return totalLoss / inputs.length;
    }

    public static void main(String[] args) {
        int inputSize = 2; // XOR input size
        int[] hiddenSizes = { 4, 4 }; // Two hidden layers with 4 neurons each
        int outputSize = 1; // XOR output size
        String[] activations_str = { "relu", "relu", "sigmoid" }; // Use sigmoid in output layer
        double learningRate = 0.1;

        NeuralNetwork nn = new NeuralNetwork(inputSize, hiddenSizes, outputSize, activations_str, learningRate);
        // for (int i = 0; i < nn.layers.size(); i++) {
        // System.out.println(Arrays.deepToString(nn.layers.get(i).weights));
        // }
        // XOR inputs and outputs
        double[][] inputs = {
                { 0, 0 },
                { 0, 1 },
                { 1, 0 },
                { 1, 1 }
        };

        double[][] targets = {
                { 0 },
                { 1 },
                { 1 },
                { 0 }
        };

        int epochs = 1; // Train for more epochs

        // Training the model
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                nn.backpropagate(inputs[i], targets[i]);
            }
            // if (epoch % 1000 == 0) {
            // double loss = nn.calculateLoss(inputs, targets);
            // System.out.println("Epoch " + epoch + " - Loss: " + loss);
            // }
        }

        // for (int i = 0; i < nn.layers.size(); i++) {
        // System.out.println(Arrays.deepToString(nn.layers.get(i).weights));
        // }

        // Testing the model
        for (double[] input : inputs) {
            List<double[]> activations = nn.forward(input);
            double[] output = activations.get(activations.size() - 1);
            System.out.println("Input: " + Arrays.toString(input) + " -> Output: " + Arrays.toString(output));
        }
    }
}
