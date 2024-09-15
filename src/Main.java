public class Main {
    public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork(2, 3, 1); // 2 inputs, 1 hidden layer with 3 neurons, 1 output

        double[][] inputs = {
                { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 }
        };

        double[][] outputs = {
                { 0 }, { 1 }, { 1 }, { 0 } // XOR problem
        };

        // Train the network
        for (int epoch = 0; epoch < 100000; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                nn.backpropagate(inputs[i], outputs[i], 0.1);
            }
        }

        // Test the network
        for (int i = 0; i < inputs.length; i++) {
            double[] output = nn.feedForward(inputs[i]);
            System.out.println("Input: " + inputs[i][0] + ", " + inputs[i][1] + " Output: " + output[0]);
        }
    }
}
