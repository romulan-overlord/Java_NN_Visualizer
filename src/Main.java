import java.io.IOException;

public class Main {
    public static void main(String[] args) {
        try {
            // Step 1: Load the Iris dataset
            IrisDataPreparation.loadIrisData("D:\\Study\\MSC\\sem3\\NN\\Neural\\Iris.csv");

            // Step 2: Get the prepared input features and labels
            double[][] inputFeatures = IrisDataPreparation.features.toArray(new double[0][0]);
            double[][] targetLabels = IrisDataPreparation.labels.toArray(new double[0][0]);

            // Step 3: Initialize the Neural Network
            // The neural network has 4 inputs (for the 4 features), 1 hidden layer with 5
            // neurons, and 3 output neurons (for 3 classes)
            NeuralNetwork nn = new NeuralNetwork(4, 5, 3);

            // Step 4: Train the Neural Network
            int epochs = 5000; // Number of training epochs
            double learningRate = 0.1; // Learning rate for backpropagation
            System.out.println("Training the neural network...");

            for (int epoch = 0; epoch < epochs; epoch++) {
                for (int i = 0; i < inputFeatures.length; i++) {
                    nn.backpropagate(inputFeatures[i], targetLabels[i], learningRate);
                }

                if (epoch % 1000 == 0) {
                    System.out.println("Epoch " + epoch + " completed.");
                }
            }

            // Step 5: Test the Neural Network
            System.out.println("Testing the neural network...");

            int correctPredictions = 0;
            for (int i = 0; i < inputFeatures.length; i++) {
                double[] output = nn.feedForward(inputFeatures[i]);

                // Get the predicted class by finding the index of the maximum output value
                int predictedClass = getPredictedClass(output);
                int actualClass = getActualClass(targetLabels[i]);

                if (predictedClass == actualClass) {
                    correctPredictions++;
                }

                System.out.println("Predicted: " + predictedClass + ", Actual: " + actualClass);
            }

            // Step 6: Calculate and print accuracy
            double accuracy = (double) correctPredictions / inputFeatures.length * 100;
            System.out.println("Accuracy: " + accuracy + "%");

            NeuralNetworkVisualizer visualizer = new NeuralNetworkVisualizer(nn);

            // Save the generated HTML to a file
            try {
                visualizer.saveHtmlToFile("neural_network_visualization.html");
                System.out.println("HTML file generated successfully!");
            } catch (IOException e) {
                e.printStackTrace();
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Helper function to get the index of the maximum output (predicted class)
    public static int getPredictedClass(double[] output) {
        int predictedClass = 0;
        double maxOutput = output[0];

        for (int i = 1; i < output.length; i++) {
            if (output[i] > maxOutput) {
                maxOutput = output[i];
                predictedClass = i;
            }
        }
        return predictedClass;
    }

    // Helper function to get the index of the actual class (from the one-hot
    // encoded target label)
    public static int getActualClass(double[] target) {
        for (int i = 0; i < target.length; i++) {
            if (target[i] == 1.0) {
                return i;
            }
        }
        return -1; // Default case (shouldn't happen if labels are correct)
    }
}
