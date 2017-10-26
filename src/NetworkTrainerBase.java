import java.util.List;

@SuppressWarnings("WeakerAccess")
public class NetworkTrainerBase implements INetworkTrainer {

    @Override
    public INeuralNetwork train(INeuralNetwork network, List<Sample> samples) {
        return null;
    }

    // Take a network and a set of inputs, run them through the network and return the outputs
    protected double[] execute(INeuralNetwork network, double[] inputs) {
        return network.execute(inputs);
    }

    // Compute the normalized squared error between a set of outputs and their true values
    protected double calculateTotalError(double[] networkOutputs, double[] expectedOutputs) {
        assert networkOutputs.length == expectedOutputs.length;

        double errorSum = 0.0;
        // Calculate the sum over the squared error for each output value
        for (int i = 0; i < networkOutputs.length; i++) {
            double error = networkOutputs[i] - expectedOutputs[i];
            errorSum += Math.pow(error, 2);
        }

        // Normalize and return error
        return errorSum / (networkOutputs.length * expectedOutputs.length);
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
