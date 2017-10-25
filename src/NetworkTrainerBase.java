import java.util.List;

public class NetworkTrainerBase implements INeuralNetworkTrainer {

    @Override
    public INeuralNetwork train(INeuralNetwork network, List<Sample> samples) {
        return null;
    }

    // Take a network and a set of inputs, run them through the network and return the outputs
    protected double[] execute(INeuralNetwork network, double[] inputs) {
        return network.execute(inputs);
    }

    // Extract the weight matrix from the network, maybe do the actual work on the network class rather than here?
    protected double[][] serializeNetwork(INeuralNetwork network) {
        return new double[0][0];
    }

    // Given a weight matrix, instantiate the network so it can be executed..
    // maybe we can encode our network execution differently and make this easier?
    protected INeuralNetwork deserializeNetwork(double[][] weightMatrix) {
        return new MultiLayerPerceptron(null, null);
    }
}
