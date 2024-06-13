import math.activation_functions.ActivationFunction;
import network.Layer;
import org.apache.commons.math3.linear.RealVector;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.Mockito;
import perceptron.Perceptron;
import perceptron.PerceptronParams;
import perceptron.Predictor;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.mockito.Mockito.*;

public class LayerTest {
    private ActivationFunction activationFunction;
    private RealVector input;
    private RealVector output;
    private RealVector error;

    @BeforeEach
    public void setup() {
        activationFunction = Mockito.mock(ActivationFunction.class);
        input = Mockito.mock(RealVector.class);
        output = Mockito.mock(RealVector.class);
        error = Mockito.mock(RealVector.class);

        when(input.getDimension()).thenReturn(6);
    }


    @Test
    public void layerInitializationCreatesCorrectNumberOfPerceptrons() {
        Layer layer = new Layer(5, activationFunction);
        assertEquals(5, layer.getPerceptrons().size());
    }

    @Test
    public void predictReturnsCorrectNumberOfOutputs() {
        Layer layer = new Layer(3, activationFunction);
        List<RealVector> outputs = layer.predict(input);

        assertEquals(3, outputs.size());
    }

    @Test
    public void trainUpdatesPerceptronParameters() {
        Layer layer = new Layer(2, activationFunction);
        List<RealVector> errors = Arrays.asList(error, error);

        // Mock the PerceptronParams and Predictor classes
        PerceptronParams params = Mockito.mock(PerceptronParams.class);
        Predictor predictor = Mockito.mock(Predictor.class);

        // Define the weights of the PerceptronParams
        double[] weights = new double[PerceptronParams.INPUT_SIZE * PerceptronParams.OUTPUT_SIZE];
        when(params.getWeights()).thenReturn(weights);

        // Stub the getParams method of the Predictor class
        when(predictor.getParams()).thenReturn(params);

        // Replace the perceptrons in the layer with the mocked Predictor
        List<Perceptron> perceptrons = Arrays.asList(predictor, predictor);
        layer.getPerceptrons().clear();
        layer.getPerceptrons().addAll(perceptrons);

        // Capture the arguments passed to setWeights and setBias
        ArgumentCaptor<double[]> weightsCaptor = ArgumentCaptor.forClass(double[].class);
        ArgumentCaptor<Double> biasCaptor = ArgumentCaptor.forClass(Double.class);

        layer.train(input, errors, 0.1);

        // Verify that setWeights and setBias were called on the PerceptronParams
        verify(params, times(2)).setWeights(weightsCaptor.capture());
        verify(params, times(2)).setBias(biasCaptor.capture());

        // Now you can make assertions about the captured arguments, which are the new weights and bias
        // For example, you can check if the new weights and bias are not null or have the expected size or value
        assertNotNull(weightsCaptor.getValue());
        assertNotNull(biasCaptor.getValue());
    }

    @Test
    public void predictCallsPredictOnEachPerceptron() {
        Layer layer = new Layer(2, activationFunction);

        // Mock the Predictor class
        Predictor predictor = Mockito.mock(Predictor.class);

        // Replace the perceptrons in the layer with the mocked Predictor
        List<Perceptron> perceptrons = Arrays.asList(predictor, predictor);
        layer.getPerceptrons().clear();
        layer.getPerceptrons().addAll(perceptrons);

        layer.predict(input);

        // Verify that the predict method was called on each Perceptron
        verify(predictor, times(2)).predict(any(RealVector.class));
    }

}