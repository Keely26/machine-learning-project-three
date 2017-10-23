public class HyperbolicTanFunction implements IActivationFunction {

    @Override
    public double compute(double value) {
        return (Math.tanh(value) + 1) / 2;
    }

    @Override
    public double computeDerivative(double value) {
        return (1 - Math.pow(Math.tanh(value), 2)) / 2;
    }
}
