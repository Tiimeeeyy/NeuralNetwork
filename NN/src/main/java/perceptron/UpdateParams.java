package perceptron;

import org.apache.commons.math3.linear.RealVector;

/**
 * Utility class to update Weights and Biases for the Perceptron.
 */

public class UpdateParams {

    private UpdateParams() {
        throw new IllegalStateException("Utility classes should not be initialised!!");
    }

    public static void update(PerceptronParams params, RealVector input, RealVector error, double learningRate) {
        double[] weights = params.getWeights();
        for (int i = 0; i < weights.length; i++) {
            weights[i] += learningRate * error.getEntry(i % PerceptronParams.OUTPUT_SIZE) * input.getEntry(i / PerceptronParams.OUTPUT_SIZE);
        }
        params.setWeights(weights);

        double bias = params.getBias();
        bias += learningRate * error.getEntry(0);
        params.setBias(bias);
    }
}
