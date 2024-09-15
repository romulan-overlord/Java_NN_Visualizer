import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class NeuralNetworkVisualizer {
    private NeuralNetwork nn; // The neural network to visualize

    public NeuralNetworkVisualizer(NeuralNetwork nn) {
        this.nn = nn;
    }

    // Method to generate HTML file for visualizing the neural network
    public void generateHTML(String fileName) {
        try (FileWriter writer = new FileWriter(fileName)) {
            writer.write("<!DOCTYPE html>\n");
            writer.write("<html lang=\"en\">\n");
            writer.write("<head>\n");
            writer.write("<meta charset=\"UTF-8\">\n");
            writer.write("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n");
            writer.write("<title>Neural Network Visualization</title>\n");
            writer.write(
                    "<link rel=\"stylesheet\" href=\"https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css\">\n");
            writer.write("<script src=\"https://code.jquery.com/jquery-3.3.1.slim.min.js\"></script>\n");
            writer.write(
                    "<script src=\"https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js\"></script>\n");
            writer.write("<style>\n");
            writer.write(".neuron-button { border-radius: 50%; width: 50px; height: 50px; margin: 10px; }\n");
            writer.write(".layer-container { text-align: center; margin-top: 20px; }\n");
            writer.write("</style>\n");
            writer.write("</head>\n");
            writer.write("<body>\n");
            writer.write("<div class=\"container\">\n");
            writer.write("<h1 class=\"text-center\">Neural Network Visualization</h1>\n");

            // Generate HTML for each layer and its neurons
            List<Layer> layers = nn.layers;
            for (int l = 0; l < layers.size(); l++) {
                writer.write("<div class=\"layer-container\">\n");
                writer.write("<h2>Layer " + (l + 1) + "</h2>\n");

                Layer layer = layers.get(l);
                for (int n = 0; n < layer.outputSize; n++) {
                    String neuronId = "neuron-" + l + "-" + n;
                    writer.write(
                            "<button type=\"button\" class=\"btn btn-primary neuron-button\" data-toggle=\"modal\" data-target=\"#"
                                    + neuronId + "\">N" + (n + 1) + "</button>\n");

                    // Create modal for the neuron details
                    writer.write("<div class=\"modal fade\" id=\"" + neuronId
                            + "\" tabindex=\"-1\" role=\"dialog\" aria-labelledby=\""
                            + neuronId + "Label\" aria-hidden=\"true\">\n");
                    writer.write("<div class=\"modal-dialog\" role=\"document\">\n");
                    writer.write("<div class=\"modal-content\">\n");
                    writer.write("<div class=\"modal-header\">\n");
                    writer.write("<h5 class=\"modal-title\" id=\"" + neuronId + "Label\">Neuron " + (n + 1)
                            + " in Layer " + (l + 1) + "</h5>\n");
                    writer.write(
                            "<button type=\"button\" class=\"close\" data-dismiss=\"modal\" aria-label=\"Close\">\n");
                    writer.write("<span aria-hidden=\"true\">&times;</span>\n");
                    writer.write("</button>\n");
                    writer.write("</div>\n");

                    writer.write("<div class=\"modal-body\">\n");
                    writer.write("<p><strong>Weights:</strong></p>\n");
                    writer.write("<ul>\n");

                    for (int w = 0; w < layer.inputSize; w++) {
                        writer.write("<li>Weight[" + w + "]: " + layer.weights[n][w] + "</li>\n");
                    }

                    writer.write("</ul>\n");
                    writer.write("<p><strong>Bias:</strong> " + layer.bias[n] + "</p>\n");
                    writer.write("</div>\n");

                    writer.write("<div class=\"modal-footer\">\n");
                    writer.write(
                            "<button type=\"button\" class=\"btn btn-secondary\" data-dismiss=\"modal\">Close</button>\n");
                    writer.write("</div>\n");

                    writer.write("</div>\n");
                    writer.write("</div>\n");
                    writer.write("</div>\n");
                }

                writer.write("</div>\n");
            }

            writer.write("</div>\n");
            writer.write("</body>\n");
            writer.write("</html>\n");

            System.out.println("HTML file generated successfully: " + fileName);
        } catch (IOException e) {
            e.printStackTrace();
        }
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

        int epochs = 10000; // Train for more epochs

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

        // Generate the HTML file to visualize the neural network
        NeuralNetworkVisualizer visualizer = new NeuralNetworkVisualizer(nn);
        visualizer.generateHTML("neural_network_visualization.html");
    }
}
