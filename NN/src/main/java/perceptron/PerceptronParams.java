package perceptron;

import math.activation_functions.ActivationFunction;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;

/**
 * This class stores all parameters of the perceptron.
 */
public class PerceptronParams {
    public static final int INPUT_SIZE = 2; // Final and static, since in our use case, they won't change.
    public static final int OUTPUT_SIZE = 3;
    private double bias;
    private ActivationFunction function;
    private double[] weights;

    /**
     * Class constructor.
     * Initialises the weights array randomly based on the "He function", since for ReLU activation function.
     *
     * @param function The Activation function to be used for the perceptron.
     */
    public PerceptronParams(ActivationFunction function) {
        this.function = function;
        this.weights = new double[INPUT_SIZE * OUTPUT_SIZE];
        this.bias = 0.0;

        RandomGenerator rnd = new JDKRandomGenerator();
        double stdDev = Math.sqrt(2.0 / INPUT_SIZE);
        for (int i = 0; i < weights.length; i++) {
            weights[i] = rnd.nextGaussian() * stdDev;
        }
    }


    /**
     * Getters and setters (Java Boilerplate)
     */

    public ActivationFunction getFunction() {
        return function;
    }

    public void setFunction(ActivationFunction function) {
        this.function = function;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public double[] getWeights() {
        return weights;
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }
}
