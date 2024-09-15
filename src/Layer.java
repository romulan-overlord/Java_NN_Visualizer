import java.util.Random;

class Layer {
    public int inputSize;
    public int outputSize;
    public double[][] weights;
    public double[] bias;
    private String activation; // Store the chosen activation function

    public Layer(int inputSize, int outputSize, String activation) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.weights = new double[outputSize][inputSize]; // weights matrix: outputSize x inputSize
        this.bias = new double[outputSize];
        this.activation = activation.toLowerCase(); // Convert to lowercase to handle case-insensitivity
        initializeWeights();
    }

    // Initialize weights randomly using Gaussian distribution
    private void initializeWeights() {
        Random random = new Random();
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weights[i][j] = random.nextGaussian() * 0.1; // Initialize weights with small random values
            }
        }

        for (int i = 0; i < outputSize; i++) {
            bias[i] = random.nextGaussian() * 0.01; // Initialize biases with small random values
        }
    }

    // Sigmoid activation function
    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    // Derivative of sigmoid function
    public static double sigmoidDerivative(double x) {
        return x * (1 - x);
    }

    // ReLU activation function
    public static double relu(double x) {
        return Math.max(0, x);
    }

    // Derivative of ReLU function
    public static double reluDerivative(double x) {
        return x > 0 ? 1 : 0;
    }

    // Dynamic activation function based on the chosen activation type
    public double activate(double x) {
        switch (activation) {
            case "relu":
                return relu(x);
            case "sigmoid":
                return sigmoid(x);
            default:
                throw new IllegalArgumentException("Unknown activation function: " + activation);
        }
    }

    // Dynamic derivative of activation function based on the chosen activation type
    public double activateDerivative(double x) {
        switch (activation) {
            case "relu":
                return reluDerivative(x);
            case "sigmoid":
                return sigmoidDerivative(x);
            default:
                throw new IllegalArgumentException("Unknown activation function: " + activation);
        }
    }

    // Forward pass through this layer
    public double[] forward(double[] input) {
        double[] output = new double[outputSize];

        for (int i = 0; i < outputSize; i++) {
            double sum = 0.0;
            for (int j = 0; j < inputSize; j++) {
                sum += input[j] * weights[i][j]; // Weighted sum
            }
            sum += bias[i]; // Add bias
            output[i] = activate(sum); // Apply activation function
        }

        return output;
    }

    // Backward pass for this layer
    public double[] backward(double[] dOutput, double[] prevActivation) {
        double[] dInput = new double[inputSize];

        // Update weights and bias and calculate gradient for previous layer
        for (int i = 0; i < outputSize; i++) {
            double delta = dOutput[i] * activateDerivative(prevActivation[i]); // Calculate delta
            for (int j = 0; j < inputSize; j++) {
                dInput[j] += weights[i][j] * delta; // Backpropagate error
            }
        }

        return dInput;
    }
}
