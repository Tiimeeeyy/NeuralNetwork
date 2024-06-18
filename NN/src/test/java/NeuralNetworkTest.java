import math.activation_functions.ActivationFunction;
import math.activation_functions.ReLUFunction;
import network.Layer;
import network.NeuralNetwork;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import perceptron.Perceptron;
import perceptron.Predictor;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class NeuralNetworkTest {
    private NeuralNetwork neuralNetwork;
    private ActivationFunction activationFunction;

    @BeforeEach
    void setUp() {
        activationFunction = new ReLUFunction();
        neuralNetwork = new NeuralNetwork(3, activationFunction); // Hidden layer with 3 perceptrons
    }

    @Test
    void testConstruction() {
        assertEquals(3, neuralNetwork.getLayers().size());
        assertArrayEquals(new int[]{2, 3, 3}, neuralNetwork.getLayerSizes());


        assertEquals(3, neuralNetwork.getActivationFunctions().length);
    }

    @Test
    void testPredict() {
        RealVector input = new ArrayRealVector(new double[]{0.5, -0.3});
        RealVector output = neuralNetwork.predict(input);

        assertEquals(3, output.getDimension());

        // Example assertions for specific output values (adjust as needed based on your perceptron weights):
        assertTrue(output.getEntry(0) >= 0); // directionX should be non-negative with ReLU
        assertTrue(output.getEntry(1) >= 0); // directionY should be non-negative with ReLU
        assertTrue(output.getEntry(2) >= 0); // velocity should be non-negative with ReLU
    }

    @Test
    void testPredictEmptyInput() {
        RealVector input = new ArrayRealVector(new double[]{}); // Empty input
        assertThrows(IllegalArgumentException.class, () -> neuralNetwork.predict(input));
    }

    @Test
    void testCalculateNodeError() {
        // Setup mock data for the hidden layer (index 1)
        List<RealVector> layerOutputs = new ArrayList<>();
        layerOutputs.add(new ArrayRealVector(new double[]{0.6, 0.8, 0.2})); // Example output from hidden layer
        layerOutputs.add(new ArrayRealVector(new double[]{0.4, 0.7, 0.1})); // Example output from output layer
        int nodeIndex = 1; // Calculate error for the second node in the hidden layer
        RealVector nextLayerErrors = new ArrayRealVector(new double[]{0.1, -0.2, 0.3}); // Errors from output layer

        // Get perceptrons and weights from the output layer (index 2)
        Layer outputLayer = neuralNetwork.getLayers().get(2);
        List<Perceptron> outputPerceptrons = outputLayer.getPerceptrons();
        double[][] outputWeights = new double[outputPerceptrons.size()][];
        for (int i = 0; i < outputPerceptrons.size(); i++) {
            outputWeights[i] = ((Predictor) outputPerceptrons.get(i)).getParams().getWeights();
        }

        // Calculate expected node error manually (based on your backpropagation logic and ReLU derivative)
        double expectedError = 0;
        for (int i = 0; i < outputPerceptrons.size(); i++) {
            expectedError += outputWeights[i][nodeIndex] * nextLayerErrors.getEntry(i) * (layerOutputs.getFirst().getEntry(nodeIndex) > 0 ? 1 : 0);
        }

        // Call the method to calculate the error
        double nodeError = neuralNetwork.calculateNodeError(layerOutputs, nodeIndex, nextLayerErrors, outputPerceptrons, activationFunction);

        // Assert that the calculated error matches the expected error (with tolerance)
        assertEquals(expectedError, nodeError, 1e-6);
    }

    @Test
    void testTrain() {
        RealVector input = new ArrayRealVector(new double[]{0.5, -0.3});
        RealVector target = new ArrayRealVector(new double[]{0.8, 0.2, 0.1}); // 3D target
        double learningRate = 0.1;

        // Store initial weights and biases for comparison
        double[][][] initialWeights = new double[neuralNetwork.getLayers().size()][][];
        double[][] initialBiases = new double[neuralNetwork.getLayers().size()][];

        for (int i = 0; i < neuralNetwork.getLayers().size(); i++) {
            Layer layer = neuralNetwork.getLayers().get(i);
            List<Perceptron> perceptrons = layer.getPerceptrons();
            initialWeights[i] = new double[perceptrons.size()][];
            initialBiases[i] = new double[perceptrons.size()];
            for (int j = 0; j < perceptrons.size(); j++) {
                initialWeights[i][j] = ((Predictor) perceptrons.get(j)).getParams().getWeights().clone(); // Copy weights array
                initialBiases[i][j] = ((Predictor) perceptrons.get(j)).getParams().getBias();
            }
        }

        neuralNetwork.train(input, target, learningRate);

        // Assert that weights and biases have been updated (use tolerances for floating-point comparisons)
        for (int i = 0; i < neuralNetwork.getLayers().size(); i++) {
            Layer layer = neuralNetwork.getLayers().get(i);
            for (int j = 0; j < layer.getPerceptrons().size(); j++) {
                assertFalse(new ArrayRealVector(initialWeights[i][j]).equals(((Predictor) layer.getPerceptrons().get(j)).getParams().getWeights()));
                assertNotEquals(initialBiases[i][j], ((Predictor) layer.getPerceptrons().get(j)).getParams().getBias(), 1e-6);
            }
        }
    }

}


