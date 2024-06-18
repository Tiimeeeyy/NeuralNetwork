package perceptron;

import org.apache.commons.math3.linear.RealVector;

/**
 * Utility class to update Weights and Biases for the Perceptron.
 */

public class UpdateParams {
    /**
     * Class constructor. Throw an exception when a class is initialized.
     */
    private UpdateParams() {
        throw new IllegalStateException("Utility classes should not be initialised!!");
    }

    /**
     * Updates the Perceptrons parameters based on Error, Learning Rate and Input Vector.
     * @param params The parameters to be updated.
     * @param input The inputted vector.
     * @param error The error with which the parameters are to be updated.
     * @param learningRate The learning rate of the System.
     */
    public static void update(PerceptronParams params, RealVector input, RealVector error, double learningRate) {
        int size = params.getSize();
        double[] weights = params.getWeights();
        for (int i = 0; i < weights.length; i++) {
            weights[i] += learningRate * error.getEntry(i % size) * input.getEntry(i / size);
        }
        params.setWeights(weights);

        double bias = params.getBias();
        for (int i = 0; i < error.getDimension(); i++) { // Assuming error vector has same size as output
            bias += learningRate * error.getEntry(i);
        }
        params.setBias(bias);
    }
}
