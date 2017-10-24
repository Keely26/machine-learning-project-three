import java.util.ArrayList;
import java.util.List;

public class MultiLayerPerceptron implements INeuralNetwork {

    private List<Layer> network;

    private double learningRate;
    private double momentum;

    private int batchSize;
    private int epochs;

    private IActivationFunction activationFunction;
    private INeuralNetworkTrainer trainer;

    private int numInputs;
    private int numOutputs;

    public MultiLayerPerceptron(int[] networkDimensions, double learningRate, int batchSize, double momentum, IActivationFunction activationFunction, int epochs) {
        this.numInputs = networkDimensions[0];
        this.numOutputs = networkDimensions[networkDimensions.length - 1];
        this.learningRate = learningRate;
        this.batchSize = batchSize;
        this.momentum = momentum;
        this.epochs = epochs;
        this.activationFunction = activationFunction;

        this.initializeNetwork(networkDimensions);
    }

    @Override
    public void train(List<Sample> samples) {
        trainer.train(this.serialize());
    }

    @Override
    public double[] approximate(double[] inputs) {
        // Forward propagate inputs through the network returning the output of the final layer
        return this.forwardPropagate(inputs);
    }

    @Override
    public double[][] serialize() {
        // TODO: Implement serialization
        return new double[0][];
    }

    @Override
    public void deserialize(double[][] weightMatrix) {
        // TODO: Implement deserialization
    }

    private double[] forwardPropagate(double[] inputs) {
        double[] layerOutputs = inputs;

        // Propagate through hidden layers, not using the activation function for the final layer
        for (int i = 0; i < network.size(); i++) {
            boolean shouldActivate = (i != network.size() - 1);
            layerOutputs = network.get(i).execute(layerOutputs, shouldActivate);
        }

        return layerOutputs;
    }


    // Initialize each layer of the network, in the input layer is created implicitly in execution and does not have a layer
    private void initializeNetwork(int[] networkDimensions) {
        this.network = new ArrayList<>(networkDimensions.length);
        for (int i = 1; i < networkDimensions.length; i++) {
            this.network.add(new Layer(networkDimensions[i], networkDimensions[i - 1], this.activationFunction));
        }
    }
}