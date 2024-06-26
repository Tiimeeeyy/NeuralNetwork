package perceptron;

import math.activation_functions.ActivationFunction;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;

import java.io.Serializable;

/**
 * This class stores all parameters of the perceptron.
 */
public class PerceptronParams implements Serializable {

    private double bias;
    private int size;
    private ActivationFunction function;
    private double[] weights;

    /**
     * Class constructor.
     * Initialises the weights array randomly based on the "He function" for ReLU activation function.
     *
     * @param function The Activation function to be used for the perceptron.
     */
    public PerceptronParams(ActivationFunction function, int size) {
        this.size = size;
        this.function = function;
        this.weights = new double[size];
        this.bias = 0.0;
        RandomGenerator rnd = new JDKRandomGenerator();
        double stdDev = Math.sqrt(2.0 / size);
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

    public int getSize() {
        return size;
    }

}
