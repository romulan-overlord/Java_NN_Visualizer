import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

public class NeuralNetworkVisualizer {

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
        // Example usage:
        int[] hiddenLayers = { 4, 4 }; // Two hidden layers with 4 neurons each
        NeuralNetwork nn = new NeuralNetwork(2, hiddenLayers, 1);

        // Generate HTML
        String htmlContent = generateHTML(nn);

        // Output to a file
        try {
            saveToFile("network_visualization.html", htmlContent);
            System.out.println("HTML file created: network_visualization.html");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
