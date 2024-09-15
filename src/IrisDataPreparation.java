import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class IrisDataPreparation {
    // This will hold the features
    public static List<double[]> features = new ArrayList<>();
    // This will hold the one-hot encoded labels
    public static List<double[]> labels = new ArrayList<>();

    public static void loadIrisData(String filePath) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(filePath));
        String line;
        boolean header = true;

        while ((line = br.readLine()) != null) {
            if (header) {
                header = false; // Skip header
                continue;
            }

            String[] data = line.split(",");

            // Parse the feature values (sepal length, sepal width, petal length, petal
            // width)
            double[] featureRow = new double[4];
            featureRow[0] = Double.parseDouble(data[1]); // SepalLengthCm
            featureRow[1] = Double.parseDouble(data[2]); // SepalWidthCm
            featureRow[2] = Double.parseDouble(data[3]); // PetalLengthCm
            featureRow[3] = Double.parseDouble(data[4]); // PetalWidthCm

            // Add the features to the list
            features.add(featureRow);

            // Parse the species and create one-hot encoded labels
            double[] labelRow = new double[3];
            String species = data[5];
            switch (species) {
                case "Iris-setosa":
                    labelRow[0] = 1;
                    labelRow[1] = 0;
                    labelRow[2] = 0;
                    break;
                case "Iris-versicolor":
                    labelRow[0] = 0;
                    labelRow[1] = 1;
                    labelRow[2] = 0;
                    break;
                case "Iris-virginica":
                    labelRow[0] = 0;
                    labelRow[1] = 0;
                    labelRow[2] = 1;
                    break;
            }

            // Add the labels to the list
            labels.add(labelRow);
        }
        br.close();

        // Normalize the features for better training
        normalizeFeatures();
    }

    // Normalize the feature values between 0 and 1
    private static void normalizeFeatures() {
        int numFeatures = 4;
        double[] minValues = new double[numFeatures];
        double[] maxValues = new double[numFeatures];

        // Initialize min and max values
        for (int i = 0; i < numFeatures; i++) {
            minValues[i] = Double.MAX_VALUE;
            maxValues[i] = Double.MIN_VALUE;
        }

        // Find min and max for each feature
        for (double[] row : features) {
            for (int i = 0; i < numFeatures; i++) {
                if (row[i] < minValues[i])
                    minValues[i] = row[i];
                if (row[i] > maxValues[i])
                    maxValues[i] = row[i];
            }
        }

        // Normalize each feature
        for (double[] row : features) {
            for (int i = 0; i < numFeatures; i++) {
                row[i] = (row[i] - minValues[i]) / (maxValues[i] - minValues[i]);
            }
        }
    }

    public static void main(String[] args) {
        try {
            // Load the Iris data from CSV
            loadIrisData("D:\\Study\\MSC\\sem3\\NN\\Neural\\Iris.csv");

            // Convert the lists into arrays for neural network usage
            double[][] inputFeatures = features.toArray(new double[0][0]);
            double[][] targetLabels = labels.toArray(new double[0][0]);

            // Example: Print some of the normalized data and labels
            for (int i = 0; i < 5; i++) {
                System.out.println("Features: " + java.util.Arrays.toString(inputFeatures[i]));
                System.out.println("Labels: " + java.util.Arrays.toString(targetLabels[i]));
            }

            // At this point, you have `inputFeatures` and `targetLabels` ready for training
            // Pass these to your NeuralNetwork class for training.

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
