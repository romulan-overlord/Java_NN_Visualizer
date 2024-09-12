import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {

    private int inputSize;
    private int outputSize;
    public List<Layer> layers; // A list of layers in the network
    private double learningRate = 0.01;

    public NeuralNetwork(int inputSize, int[] hiddenSizes, int outputSize, String[] activations) {
        if (hiddenSizes.length != activations.length) {
            throw new IllegalArgumentException("Number of hidden layers and activations must match.");
        }

        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.layers = new ArrayList<>();

        // Create hidden layers with specified activation functions
        int previousSize = inputSize;
        for (int i = 0; i < hiddenSizes.length; i++) {
            layers.add(new Layer(previousSize, hiddenSizes[i], activations[i]));
            previousSize = hiddenSizes[i];
        }

        // Output layer (assume sigmoid activation for the output layer, adjust as
        // needed)
        layers.add(new Layer(previousSize, outputSize, "sigmoid"));
    }

    // Forward pass through the whole network
    public double[] forward(double[] input) {
        double[] activations = input;

        for (Layer layer : layers) {
            activations = layer.forward(activations);
        }

        return activations;
    }

    // // Backpropagation
    // public void backpropagate(double[] input, double[] target) {
    // List<double[]> activations = new ArrayList<>();
    // List<double[]> weightedInputs = new ArrayList<>();

    // // Forward pass (store activations and weighted inputs for backpropagation)
    // double[] currentInput = input;
    // activations.add(currentInput);
    // for (Layer layer : layers) {
    // double[] z = new double[layer.outputSize];
    // double[] a = new double[layer.outputSize];
    // for (int i = 0; i < layer.outputSize; i++) {
    // z[i] = 0.0;
    // for (int j = 0; j < layer.inputSize; j++) {
    // z[i] += currentInput[j] * layer.weights[j][i];
    // }
    // z[i] += layer.bias[i];
    // a[i] = layer.activate(z[i]); // Use dynamic activation function
    // }
    // weightedInputs.add(z);
    // activations.add(a);
    // currentInput = a;
    // }

    // // Calculate output error (target - output)
    // double[] outputError = new double[outputSize];
    // double[] outputActivation = activations.get(activations.size() - 1);
    // for (int i = 0; i < outputSize; i++) {
    // outputError[i] = (target[i] - outputActivation[i])
    // * layers.get(layers.size() - 1).activateDerivative(outputActivation[i]);
    // }

    // // Backpropagate the error and update weights and biases
    // double[] error = outputError;
    // for (int l = layers.size() - 1; l >= 0; l--) {
    // Layer layer = layers.get(l);
    // double[] prevActivation = activations.get(l);
    // double[] newError = new double[layer.inputSize];

    // // Update weights and calculate new error for previous layer
    // for (int i = 0; i < layer.outputSize; i++) {
    // for (int j = 0; j < layer.inputSize; j++) {
    // layer.weights[j][i] += learningRate * error[i] * prevActivation[j];
    // newError[j] += error[i] * layer.weights[j][i];
    // }
    // layer.bias[i] += learningRate * error[i];
    // }

    // // Update error for previous layer (using derivative of activation function)
    // if (l > 0) {
    // for (int i = 0; i < layer.inputSize; i++) {
    // newError[i] *= layers.get(l - 1).activateDerivative(prevActivation[i]);
    // }
    // }

    // error = newError;
    // }
    // }

    public void backpropagate(double[] input, double[] target) {
        List<double[]> activations = new ArrayList<>();
        List<double[]> weightedInputs = new ArrayList<>();

        // Forward pass (store activations and weighted inputs for backpropagation)
        double[] currentInput = input;
        activations.add(currentInput);
        for (Layer layer : layers) {
            double[] z = new double[layer.outputSize];
            double[] a = new double[layer.outputSize];
            for (int i = 0; i < layer.outputSize; i++) {
                z[i] = 0.0;
                for (int j = 0; j < layer.inputSize; j++) {
                    z[i] += currentInput[j] * layer.weights[i][j]; // Corrected access
                }
                z[i] += layer.bias[i];
                a[i] = layer.activate(z[i]); // Use dynamic activation function
            }
            weightedInputs.add(z);
            activations.add(a);
            currentInput = a;
        }

        // Calculate output error (target - output)
        double[] outputError = new double[outputSize];
        double[] outputActivation = activations.get(activations.size() - 1);
        for (int i = 0; i < outputSize; i++) {
            outputError[i] = (target[i] - outputActivation[i])
                    * layers.get(layers.size() - 1).activateDerivative(outputActivation[i]);
        }

        // Backpropagate the error and update weights and biases
        double[] error = outputError;
        for (int l = layers.size() - 1; l >= 0; l--) {
            Layer layer = layers.get(l);
            double[] prevActivation = activations.get(l);
            double[] newError = new double[layer.inputSize];

            // Update weights and calculate new error for previous layer
            for (int i = 0; i < layer.outputSize; i++) {
                for (int j = 0; j < layer.inputSize; j++) {
                    layer.weights[i][j] += learningRate * error[i] * prevActivation[j]; // Corrected access
                    newError[j] += error[i] * layer.weights[i][j]; // Corrected access
                }
                layer.bias[i] += learningRate * error[i];
            }

            // Update error for previous layer (using derivative of activation function)
            if (l > 0) {
                for (int i = 0; i < layer.inputSize; i++) {
                    newError[i] *= layers.get(l - 1).activateDerivative(prevActivation[i]);
                }
            }

            error = newError;
        }
    }

    // Utility to train the network
    public void train(double[][] inputs, double[][] targets, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                backpropagate(inputs[i], targets[i]);
            }
            if (epoch % 100 == 0) {
                System.out.println("Epoch: " + epoch);
            }
        }
    }

    public static String generateHTML(NeuralNetwork nn) {
        StringBuilder html = new StringBuilder();
        List<Layer> layers = nn.layers; // Access the layers of the neural network

        // HTML Boilerplate with Bootstrap
        html.append("<!DOCTYPE html>\n");
        html.append("<html lang=\"en\">\n");
        html.append("<head>\n");
        html.append("    <meta charset=\"UTF-8\">\n");
        html.append("    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1, shrink-to-fit=no\">\n");
        html.append(
                "    <link href=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css\" rel=\"stylesheet\">\n");
        html.append("    <title>Neural Network Visualization</title>\n");
        html.append("    <style>\n");
        html.append("        .node {\n");
        html.append("            width: 50px;\n");
        html.append("            height: 50px;\n");
        html.append("            border-radius: 50%;\n");
        html.append("            background-color: #6c757d;\n");
        html.append("            color: white;\n");
        html.append("            display: flex;\n");
        html.append("            align-items: center;\n");
        html.append("            justify-content: center;\n");
        html.append("            margin: 10px;\n");
        html.append("        }\n");
        html.append("        .layer {\n");
        html.append("            display: flex;\n");
        html.append("            justify-content: center;\n");
        html.append("            align-items: center;\n");
        html.append("            margin-bottom: 30px;\n");
        html.append("        }\n");
        html.append("        .arrow {\n");
        html.append("            text-align: center;\n");
        html.append("            font-size: 20px;\n");
        html.append("            color: #000;\n");
        html.append("        }\n");
        html.append("    </style>\n");
        html.append("</head>\n");
        html.append("<body>\n");
        html.append("<div class=\"container\">\n");
        html.append("    <h1 class=\"text-center my-4\">Neural Network Structure</h1>\n");

        // Visualize each layer of the neural network
        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            html.append("    <div class=\"layer\">\n");

            // Nodes of the current layer as buttons
            for (int j = 0; j < layer.outputSize; j++) {
                String buttonId = "neuron-" + i + "-" + j; // Unique ID for each neuron
                String modalId = "modal-" + i + "-" + j; // Unique ID for each modal

                html.append(
                        "        <button type=\"button\" class=\"node btn btn-secondary\" data-bs-toggle=\"modal\" data-bs-target=\"#")
                        .append(modalId).append("\">")
                        .append(j + 1)
                        .append("</button>\n");

                // Add modal for neuron
                html.append("        <div class=\"modal fade\" id=\"").append(modalId)
                        .append("\" tabindex=\"-1\" aria-labelledby=\"").append(modalId)
                        .append("-label\" aria-hidden=\"true\">\n");
                html.append("            <div class=\"modal-dialog\">\n");
                html.append("                <div class=\"modal-content\">\n");
                html.append("                    <div class=\"modal-header\">\n");
                html.append("                        <h5 class=\"modal-title\" id=\"").append(modalId)
                        .append("-label\">Weights for Neuron ").append(j + 1).append(" (Layer ").append(i + 1)
                        .append(")</h5>\n");
                html.append(
                        "                        <button type=\"button\" class=\"btn-close\" data-bs-dismiss=\"modal\" aria-label=\"Close\"></button>\n");
                html.append("                    </div>\n");
                html.append("                    <div class=\"modal-body\">\n");

                // Show the weights for the current neuron
                html.append("                        <ul>\n");
                for (int k = 0; k < layer.inputSize; k++) {
                    html.append("                            <li>Weight ").append(k + 1).append(": ")
                            .append(layer.weights[k][j]).append("</li>\n");
                }
                html.append("                        </ul>\n");

                html.append("                    </div>\n");
                html.append("                    <div class=\"modal-footer\">\n");
                html.append(
                        "                        <button type=\"button\" class=\"btn btn-secondary\" data-bs-dismiss=\"modal\">Close</button>\n");
                html.append("                    </div>\n");
                html.append("                </div>\n");
                html.append("            </div>\n");
                html.append("        </div>\n");
            }

            html.append("    </div>\n");

            // Arrow between layers, except for the last layer
            if (i < layers.size() - 1) {
                html.append("    <div class=\"arrow\">&#8595;</div>\n");
            }
        }

        html.append("</div>\n");
        html.append(
                "<script src=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js\"></script>\n");
        html.append("</body>\n");
        html.append("</html>\n");

        return html.toString();
    }

    // Save HTML to file
    public static void saveToFile(String filename, String content) throws IOException {
        FileWriter writer = new FileWriter(filename);
        writer.write(content);
        writer.close();
    }

    public static void main(String[] args) {
        // Example usage
        int[] hiddenLayers = { 4, 4 }; // Two hidden layers with 4 neurons each
        String[] activations = { "relu", "relu" }; // Activations for hidden layers
        NeuralNetwork nn = new NeuralNetwork(2, hiddenLayers, 1, activations);

        // XOR problem dataset
        double[][] inputs = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
        double[][] targets = { { 0 }, { 1 }, { 1 }, { 0 } };

        // Train the network
        nn.train(inputs, targets, 100);

        // Test the network
        for (double[] input : inputs) {
            double[] output = nn.forward(input);
            System.out.println("Input: " + input[0] + ", " + input[1] + " => Output: " + output[0]);
        }
    }
}
