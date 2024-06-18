package network;


import math.activation_functions.ActivationFunction;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import perceptron.Perceptron;
import perceptron.Predictor;
import perceptron.UpdateParams;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class NeuralNetwork {

    private static final int INPUT_LAYER_SIZE = 2;
    private static final int OUTPUT_LAYER_SIZE = 3;
    private final int[] layerSizes;
    private List<Layer> layers;
    private ActivationFunction[] activationFunctions;

    public NeuralNetwork(int hiddenLayerSize, ActivationFunction activationFunction) {

        this.layerSizes = new int[]{INPUT_LAYER_SIZE, hiddenLayerSize, OUTPUT_LAYER_SIZE};
        this.activationFunctions = new ActivationFunction[]{activationFunction, activationFunction, activationFunction};

        layers = new ArrayList<>();
        for (int i = 0; i < layerSizes.length; i++) {
            layers.add(new Layer(layerSizes[i], activationFunctions[i]));
        }
    }

    /**
     *
     * @param input
     * @return
     */
    public RealVector predict(RealVector input) {
        RealVector output = input;
        for (Layer layer : layers) {
            output = layer.predict(output);
        }
        return output;
    }

    /**
     * Calculated the Error of the Node based on the Following parameters.
     * @param layerOutputs The outputs of all layers.
     * @param nodeIndex The index of the Node we're accessing.
     * @param nextLayerErrors The Errors of the next node.
     * @param prevLayerPerceptron The Perceptron of the previous layer.
     * @param activationFunction The activation function of the System.
     * @return The Node error (double).
     */
    public double calculateNodeError(List<RealVector> layerOutputs, int nodeIndex, RealVector nextLayerErrors, List<Perceptron> prevLayerPerceptron, ActivationFunction activationFunction) {
        if (nodeIndex >= layerOutputs.size()) {
            throw new IllegalArgumentException("Node index is bigger than layerOutputs: " + nodeIndex + " " + layerOutputs.size());
        }
        if (nextLayerErrors.getDimension() > prevLayerPerceptron.size()) {
            throw new IllegalArgumentException("Next layers error vector dimension is greater than previous layer!" + nextLayerErrors.getDimension() + " " + prevLayerPerceptron.size());
        }

        double error = 0;

        for (int i = 0; i < nextLayerErrors.getDimension(); i++) {

            RealVector perceptronWeights = new ArrayRealVector(((Predictor) prevLayerPerceptron.get(i)).getParams().getWeights());
            double weight = perceptronWeights.getEntry(nodeIndex);
            error += weight * nextLayerErrors.getEntry(i);
        }

        double temp = layerOutputs.getFirst().getEntry(nodeIndex);
        double activated = activationFunction.deriv(temp);
        return error * activated;

    }

    public void train(RealVector input, RealVector target, double learningRate) {
        List<RealVector> layerInputs = new ArrayList<>();
        layerInputs.add(input);
        List<RealVector> layerOutputs = new ArrayList<>();

        for (Layer layer : layers) {
            RealVector output = layer.predict(layerInputs.getLast());
            layerOutputs.add(output);
            layerInputs.add(output);
        }

        backpropagate(layerInputs, layerOutputs, target, learningRate);

    }

    private void backpropagate(List<RealVector> layerInputs, List<RealVector> layerOutputs, RealVector target, double learningRate) {
        List<RealVector> errors = new ArrayList<>();
        errors.add(target.subtract(layerOutputs.getLast()));

        for (int i = layers.size() - 1; i > 0; i--) {

            Layer currentLayer = layers.get(i);
            Layer prevLayer = layers.get(i - 1);

            List<RealVector> currentLayerOutputs = Collections.singletonList(layerOutputs.get(i));
            int size = Math.min(currentLayer.getPerceptrons().size(), currentLayerOutputs.size());
            List<Double> currentLayerErrors = new ArrayList<>(Collections.nCopies(size, 0.0));
            for (int j = 0; j < size; j++) {
              double error = calculateNodeError(currentLayerOutputs, j, errors.getFirst(), prevLayer.getPerceptrons(), activationFunctions[i - 1]);
                currentLayerErrors.set(j,error);
            }
            for (int j = 0; j < prevLayer.getPerceptrons().size() - 1; j++) {
                System.out.println("LAYERS UPDATED");
                Perceptron perceptron = prevLayer.getPerceptrons().get(j);
                double perceptronErrorDouble = currentLayerErrors.get(j);
                double[] perceptronErrorArray = new double[((Predictor)perceptron).getParams().getWeights().length];
                Arrays.fill(perceptronErrorArray, perceptronErrorDouble);
                RealVector perceptronError = new ArrayRealVector(perceptronErrorArray);
                System.out.println("Params " + Arrays.toString(((Predictor) perceptron).getParams().getWeights()));
                System.out.println("Layer Inputs " + layerInputs.get(i-1));
                System.out.println("Error " + perceptronError);
                System.out.println("Learning rate " + learningRate);
                UpdateParams.update(
                        ((Predictor) perceptron).getParams(),
                        layerInputs.get(i - 1),
                        perceptronError,
                        learningRate
                );
            }
            errors.removeFirst();
            if (i > 1) {
                double[] combinedErrors = currentLayerErrors.stream()
                        .mapToDouble(Double::doubleValue)
                        .toArray();
                errors.addFirst(new ArrayRealVector(combinedErrors));

            }
        }
    }

    /**
     * Getters and setters.
     */
    public ActivationFunction[] getActivationFunctions() {
        return activationFunctions;
    }

    public int[] getLayerSizes() {
        return layerSizes;
    }

    public List<Layer> getLayers() {
        return layers;
    }
}


