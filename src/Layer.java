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
        this.weights = new double[outputSize][inputSize]; // might need to be reversed
        this.bias = new double[outputSize];
        this.activation = activation.toLowerCase(); // Convert to lowercase to handle case-insensitivity
        initializeWeights();
    }

    // Initialize weights randomly
    private void initializeWeights() {
        Random random = new Random();
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weights[i][j] = random.nextGaussian() * 0.01;
            }
        }

        for (int i = 0; i < outputSize; i++) {
            bias[i] = random.nextGaussian() * 0.01;
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

    // Choose the activation function dynamically
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

    // Choose the activation derivative function dynamically
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

    // Forward pass through this layer with dynamic activation
    public double[] forward(double[] input) {
        double[] output = new double[outputSize];

        for (int i = 0; i < outputSize; i++) {
            double sum = 0.0;
            for (int j = 0; j < inputSize; j++) {
                sum += input[j] * weights[i][j];
            }
            sum += bias[i];
            output[i] = activate(sum); // Use dynamic activation function
        }

        return output;
    }

    // Backpropagation (if needed, dynamic derivative for each neuron)
    public double[] backward(double[] dOutput) {
        double[] dInput = new double[inputSize];

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                dInput[i] += dOutput[j] * weights[i][j] * activateDerivative(weights[i][j]);
            }
        }

        return dInput;
    }
}
