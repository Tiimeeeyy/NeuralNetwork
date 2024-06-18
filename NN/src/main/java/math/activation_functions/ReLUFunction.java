package math.activation_functions;

public class ReLUFunction implements ActivationFunction {
    // This specific function uses the Leaky ReLU function
    // https://en.wikipedia.org/wiki/Rectifier_(neural_networks)

    public double activation(double parameter) {
        return Math.max(0.01 * parameter, parameter);
    }

    public double deriv(double paramter) {
        if (paramter > 0) {
            return 1.0;
        }
        return 0.01;
    }
}
