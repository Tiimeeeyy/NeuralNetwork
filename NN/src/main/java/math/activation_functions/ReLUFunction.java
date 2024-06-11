package math.activation_functions;

public class ReLUFunction implements ActivationFunction {

    public double activation(double parameter) {
        return Math.max(0, parameter);
    }

    public double deriv(double paramter) {
        return paramter > 0 ? 1 : 0;
    }
}
