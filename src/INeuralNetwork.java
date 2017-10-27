
public interface INeuralNetwork {
    double[] execute(double[] inputs);
    Layer getLayer(int index);
    int getSize();
    WeightMatrix constructWeightMatrix();
    double computeActivation(double input);
    double computeActivationDerivative(double input);
}
