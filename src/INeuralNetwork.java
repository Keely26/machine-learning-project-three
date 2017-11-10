
public interface INeuralNetwork {
    double[] execute(double[] inputs);
    Layer getLayer(int index);
    int getSize();
    double getConvergence();
    void setConvergence(double convergenceTime);
    WeightMatrix constructWeightMatrix();
    double computeActivation(double input);
    double computeActivationDerivative(double input);
}
