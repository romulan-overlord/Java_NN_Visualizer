import java.util.Random;

public class Neuron {
    public double[] weights;
    public double bias;
    public double output;
    public double delta;

    public Neuron(int inputSize) {
        weights = new double[inputSize];
        bias = new Random().nextDouble() - 0.5;

        // Initialize weights randomly
        Random random = new Random();
        for (int i = 0; i < weights.length; i++) {
            weights[i] = random.nextDouble() - 0.5;
        }
    }

    // Feedforward step: Apply weights and bias to input
    public double feedForward(double[] inputs) {
        double sum = bias;
        for (int i = 0; i < inputs.length; i++) {
            sum += inputs[i] * weights[i];
        }
        output = ActivationFunction.sigmoid(sum);
        return output;
    }
}
