
/**
 * Interface that enforces activation function method contracts
 * Used with both MLP and RBF allowing interchange and flexibility of different activation functions
 */
public interface IActivationFunction {
    double compute(double value);
    double computeDerivative(double value);
}
