
public class NetworkFactory {

    /* Backpropagation Parameters */
    private static final int[] layers = new int[]{9, 10, 5, 1};    // Size of each layer
    private static final int batchSize = 5;
    private static final double learningRate = 0.2;
    private static final double momentum = 0.0;
    private static final IActivationFunction activationFunction = new HyperbolicTangent();
    private static final int epochs = 50000;

    /* Evolution Strategy Parameters */
    private static final int populationSize = 100;
    private static final int numberOffspring = 10;
    private static final int numberParents = 2;
    private static final double mutationRate = 0.001;


    public static INetworkTrainer buildNetworkTrainer(NetworkTrainerType type) {
        switch (type) {
            case BPNetworkTrainer:
                return new BPNetworkTrainer(learningRate, momentum, batchSize, epochs);
            case DENetworkTrainer:
                return new DENetworkTrainer(100);
            case ESNetworkTrainer:
                return new ESNetworkTrainer(populationSize, numberParents, numberOffspring, mutationRate);
            case GANetworkTrainer:
                return new GANetworkTrainer();
            default:
                return null;
        }
    }

    // Construct a new network of the requested type
    public static INeuralNetwork buildNewNetwork(NetworkType type) {
        if (type == NetworkType.MultiLayerPerceptron) {
            return new MultiLayerPerceptron(activationFunction, layers);
        } else {
            return null;
        }
    }
}
