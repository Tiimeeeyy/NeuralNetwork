package network;

import math.activation_functions.ActivationFunction;
import org.apache.commons.math3.linear.RealVector;
import perceptron.Perceptron;
import perceptron.PerceptronParams;
import perceptron.Predictor;
import perceptron.UpdateParams;

import java.util.ArrayList;
import java.util.List;

// TODO: Input from pathfinding algorithm
// TODO: Idea: Implement a stack, check which one is the closest , if it is closest, pop from stack and get to the next elements with consecutive shots.
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

            PerceptronParams params = new PerceptronParams(activationFunction);
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
    public List<RealVector> predict(RealVector input) {
        List<RealVector> outputs = new ArrayList<>();

        for (Perceptron perceptron : perceptrons) {
            outputs.add(perceptron.predict(input));
        }
        return outputs;
    }

    public void train(RealVector input, List<RealVector> errors, double learningRate) {

        for (int i = 0; i < perceptrons.size(); i++) {

            RealVector error = errors.get(i);
            Predictor perceptron = (Predictor) perceptrons.get(i);

            UpdateParams.update(perceptron.getParams(), input, error, learningRate);
        }
    }

    public List<Perceptron> getPerceptrons() {
        return perceptrons;
    }
}
