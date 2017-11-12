import java.util.ArrayList;
import java.util.List;

public class MultiLayerPerceptron implements INeuralNetwork {

    private List<Layer> layers;
    private final IActivationFunction activationFunction;

    private double convergenceTime;

    MultiLayerPerceptron(IActivationFunction activationFunction, int[] networkDimensions) {
        this.activationFunction = activationFunction;
        this.initializeNetwork(networkDimensions);
    }

    /**
     * Execute a forward propagation through the network using the supplied array as inputs
     */
    public double[] execute(double[] inputs) {
        if (inputs.length != this.layers.get(0).getNeuron(0).getWeights().size()) {
            throw new IllegalArgumentException("Input/Network size mismatch!");
        }

        for (int i = 0; i < this.layers.size(); i++) {
            inputs = layers.get(i).execute(inputs, i != this.layers.size() - 1);
        }

        return inputs;
    }

    public Layer getLayer(int index) {
        return this.layers.get(index);
    }

    @Override
    public WeightMatrix constructWeightMatrix() {
        return new WeightMatrix(this);
    }

    @Override
    public int getSize() {
        return this.layers.size();
    }

    @Override
    public double getConvergence() {
        return this.convergenceTime;
    }

    @Override
    public void setConvergence(double convergenceTime) {
        this.convergenceTime = convergenceTime;
    }

    public double computeActivationDerivative(double input) {
        return this.activationFunction.computeDerivative(input);
    }

    /**
     * Create a new network of the supplied dimensions
     */
    private void initializeNetwork(int[] networkDimensions) {
        if (networkDimensions == null || networkDimensions.length < 2) {
            throw new IllegalArgumentException("Invalid network configuration!");
        }

        this.layers = new ArrayList<>(networkDimensions.length);
        for (int i = 1; i < networkDimensions.length; i++) {
            this.layers.add(new Layer(networkDimensions[i], networkDimensions[i - 1], this.activationFunction));
        }
    }
}