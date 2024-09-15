import java.io.FileWriter;
import java.io.IOException;

public class NeuralNetworkVisualizer {

    private NeuralNetwork network;

    public NeuralNetworkVisualizer(NeuralNetwork network) {
        this.network = network;
    }

    // Generate the HTML code to visualize the network architecture
    public String generateHtml() {
        StringBuilder html = new StringBuilder();

        html.append("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n")
                .append("<meta charset=\"UTF-8\">\n")
                .append("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n")
                .append("<title>Neural Network Visualizer</title>\n")
                .append("<link rel=\"stylesheet\" href=\"https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css\">\n")
                .append("<style>\n")
                .append(".layer { display: inline-block; margin: 20px; }\n")
                .append(".neuron { background-color: #6c757d; border-radius: 50%; width: 50px; height: 50px; display: flex; align-items: center; justify-content: center; margin: 10px auto; color: white; }\n")
                .append(".input-neuron { background-color: lightgreen; }\n")
                .append(".hidden-neuron { background-color: #007bff; }\n")
                .append(".output-neuron { background-color: lightcoral; }\n")
                .append("svg { position: absolute; }\n")
                .append("</style>\n</head>\n<body>\n")
                .append("<div class=\"container text-center\">\n")
                .append("<h2>Neural Network Architecture</h2>\n<div style=\"position:relative;\">\n");

        // Generate layers with neurons
        int inputLayerSize = network.layers[0].neurons.length;
        html.append(generateLayerHtml("Input layer", inputLayerSize, "input-neuron"));
        for (int i = 0; i < network.layers.length - 1; i++) {
            int hiddenLayerSize = network.layers[i].neurons.length;
            html.append(generateLayerHtml("Hidden layer " + (i + 1), hiddenLayerSize, "hidden-neuron"));
        }
        int outputLayerSize = network.layers[network.layers.length - 1].neurons.length;
        html.append(generateLayerHtml("Output layer", outputLayerSize, "output-neuron"));

        html.append("</div>\n")
                .append(generateConnections())
                .append("</div>\n")
                .append("<script src=\"https://code.jquery.com/jquery-3.2.1.slim.min.js\"></script>\n")
                .append("<script src=\"https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.11.0/umd/popper.min.js\"></script>\n")
                .append("<script src=\"https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js\"></script>\n")
                .append("</body>\n</html>");

        return html.toString();
    }

    // Helper function to generate the HTML for a layer
    private String generateLayerHtml(String layerName, int numNeurons, String neuronClass) {
        StringBuilder html = new StringBuilder();
        html.append("<div class=\"layer\">\n<h4>").append(layerName).append("</h4>\n");
        for (int i = 0; i < numNeurons; i++) {
            html.append("<div class=\"neuron ").append(neuronClass).append("\">").append(i + 1).append("</div>\n");
        }
        html.append("</div>\n");
        return html.toString();
    }

    // Helper function to generate SVG lines between neurons to represent
    // connections
    private String generateConnections() {
        StringBuilder svg = new StringBuilder();
        svg.append("<svg width=\"1000\" height=\"600\">\n");

        // Assuming fixed positions for neurons and layers for simplicity
        int[][] neuronPositions = {
                { 100, 100 }, { 100, 200 }, { 100, 300 }, { 100, 400 }, // Input neurons
                { 300, 150 }, { 300, 250 }, { 300, 350 }, { 300, 450 }, // Hidden layer neurons
                { 500, 250 } // Output neuron
        };

        // Connections from input to hidden layer
        for (int i = 0; i < 4; i++) {
            for (int j = 4; j < 8; j++) {
                svg.append("<line x1=\"").append(neuronPositions[i][0])
                        .append("\" y1=\"").append(neuronPositions[i][1])
                        .append("\" x2=\"").append(neuronPositions[j][0])
                        .append("\" y2=\"").append(neuronPositions[j][1])
                        .append("\" stroke=\"black\" />\n");
            }
        }

        // Connections from hidden layer to output neuron
        for (int i = 4; i < 8; i++) {
            svg.append("<line x1=\"").append(neuronPositions[i][0])
                    .append("\" y1=\"").append(neuronPositions[i][1])
                    .append("\" x2=\"").append(neuronPositions[8][0])
                    .append("\" y2=\"").append(neuronPositions[8][1])
                    .append("\" stroke=\"black\" />\n");
        }

        svg.append("</svg>\n");
        return svg.toString();
    }

    // Save the HTML to a file
    public void saveHtmlToFile(String filename) throws IOException {
        String htmlContent = generateHtml();
        FileWriter writer = new FileWriter(filename);
        writer.write(htmlContent);
        writer.close();
    }
}
