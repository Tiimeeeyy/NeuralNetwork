package network;

import math.activation_functions.ActivationFunction;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import perceptron.Perceptron;
import perceptron.PerceptronParams;
import perceptron.Predictor;
import perceptron.UpdateParams;

import java.util.ArrayList;
import java.util.List;

public class Layer {
    private List<Perceptron> perceptrons;

    /**
     * Creates a Layer for the Neural Network.
     *
     * @param numPerceptrons     The number of perceptron in the layer.
     * @param activationFunction The activation function used in the layer.
     */
    public Layer(int numPerceptrons, ActivationFunction activationFunction) {
        perceptrons = new ArrayList<>();
        for (int i = 0; i < numPerceptrons; i++) {

            PerceptronParams params = new PerceptronParams(activationFunction, numPerceptrons);
            perceptrons.add(new Predictor(params));

        }
    }

    /**
     * The predict method for the layer. It takes each perceptron in that layer and predicts its
     * outcome.
     *
     * @param input The input.
     * @return A list of Vectors containing the predicted outputs.
     */
    public RealVector predict(RealVector input) {
        RealVector outputs = new ArrayRealVector(3);
        int i = 0;
        for (Perceptron perceptron : perceptrons) {
            RealVector prediction = perceptron.predict(input);
            if (prediction.getDimension() != outputs.getDimension()) {
                throw new IllegalArgumentException("Dimension Mismatch: Prediction size: " + prediction.getDimension() + " output size: " + outputs.getDimension());
            }
            outputs.add(perceptron.predict(input));
        }
        return outputs;
    }

    /**
     * Trains the Specific layer based on input data.
     * @param input The input data in Vector Form.
     * @param errors The List of Errors Produced.
     * @param learningRate The learning rate.
     */
    public void train(RealVector input, List<RealVector> errors, double learningRate) {

        for (int i = 0; i < perceptrons.size(); i++) {

            RealVector error = errors.get(i);
            Predictor perceptron = (Predictor) perceptrons.get(i);

            //UpdateParams.update(perceptron.getParams(), input, error, learningRate);
        }
    }

    /**
     * Gets the List of perceptrons
     * @return List of perceptrons.
     */
    public List<Perceptron> getPerceptrons() {
        return perceptrons;
    }
}
