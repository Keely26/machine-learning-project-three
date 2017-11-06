import java.util.stream.IntStream;

public class NetworkTrainerBase implements INetworkTrainer {

    @Override
    public INeuralNetwork train(INeuralNetwork network, Dataset samples) {
        System.out.println("Train should be called an instance of the base, not the base class itself!!");
        System.exit(-1);
        return null;
    }

    // Take a network and a set of inputs, run them through the network and return the outputs
    protected double[] execute(INeuralNetwork network, double[] inputs) {
        return network.execute(inputs);
    }

    protected void evaluateFitness(Population population, Dataset trainingData) {
        population.forEach((WeightMatrix individual) -> {
            double fitness = trainingData
                    .stream()
                    .mapToDouble((Sample sample) -> {
                        INeuralNetwork network = individual.buildNetwork();
                        double[] networkOutputs = network.execute(sample.inputs);
                        return meanSquaredError(networkOutputs, sample.outputs);
                    })
                    .sum();
            individual.setFitness(fitness);
        });
    }

    // Compute the normalized squared error between a set of outputs and their true values
    protected double meanSquaredError(double[] networkOutputs, double[] expectedOutputs) {
        assert networkOutputs.length == expectedOutputs.length;

        // Calculate the sum over the squared error for each output value
        double errorSum = IntStream.range(0, networkOutputs.length)
                .mapToDouble(i -> Math.pow(networkOutputs[i] - expectedOutputs[i], 2))
                .sum();

        // Normalize and return error
        return errorSum / (networkOutputs.length * expectedOutputs.length);
    }

    /**
     * Convert network into a new weight matrix containing a 1D array of weights
     */
    protected WeightMatrix serializeNetwork(INeuralNetwork network) {
        return new WeightMatrix(network);
    }

    /**
     * Return the network represented by the provided weight matrix
     */
    protected INeuralNetwork deserializeNetwork(WeightMatrix weightMatrix) {
        return weightMatrix.buildNetwork();
    }
}
