package perceptron;

import math.activation_functions.ActivationFunction;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

/**
 * Predictor class implements Perceptron.
 * This class is used to predict using the perceptron.
 */
public class Predictor implements Perceptron {

    private PerceptronParams params;

    /**
     * Class constructor.
     *
     * @param params The parameters for the perceptron.
     */
    public Predictor(PerceptronParams params) {
        this.params = params;
    }

    /**
     * Predicts an output (in our case, a hit to the golf ball) based on the Input given.
     *
     * @param input The current state of the Game (Ball position)
     * @return An Array that contains the prediction.
     */
    public RealVector predict(RealVector input) {
        double[] weights = params.getWeights();
        ActivationFunction function = params.getFunction();
        double bias = params.getBias();
        RealVector weightVector = new ArrayRealVector(weights);
        double dotProduct = weightVector.dotProduct(input);
        double[] result = new double[weights.length / input.getDimension()];

        for (int i = 0; i < result.length; i++) {
            result[i] = function.activation(dotProduct + bias);
        }
        return new ArrayRealVector(result);
    }

    @Override
    public void train(RealVector input, RealVector target, double learningRate) { /* Empty method to allow polymorphism */ }

    public PerceptronParams getParams() {
        return params;
    }
}
