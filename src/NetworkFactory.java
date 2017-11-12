
public class NetworkFactory {

    /* MultiLayer Perceptron Parameters */
    private static final int[] layers = new int[]{11, 15, 1};    // Size of each layer
    private static final IActivationFunction activationFunction = new HyperbolicTangent();

    /* Backpropagation Parameters */
    private static final int batchSize = 3;
    private static final double learningRate = 0.01;
    private static final double momentum = 0.0;

    /* Evolution Strategy Parameters */
    private static final int populationSizeES = 75;
    private static final int numberOffspring = 10;
    private static final int numberParents = 5;
    private static final double mutationRate = 0.01;

    /* Differential Evolution Parameters */
    private static final int populationSizeDE = 150;
    private static final double beta = 0.6;
    private static final double crossoverRate = 0.05;

    /* Genetic Algorithm Parameters */
    private static final int populationSizeGA = 75;
    private static final double mutationRateGA = 0.001;
    private static final int numParentsGA = 5;
    private static final int numberOffspringGA = 10;


    public static INetworkTrainer buildNetworkTrainer(NetworkTrainerType type) {
        switch (type) {
            case BPNetworkTrainer:
                return new BPNetworkTrainer(learningRate, momentum, batchSize);
            case DENetworkTrainer:
                return new DENetworkTrainer(populationSizeDE, beta, crossoverRate);
            case ESNetworkTrainer:
                return new ESNetworkTrainer(populationSizeES, numberParents, numberOffspring, mutationRate);
            case GANetworkTrainer:
                return new GANetworkTrainer(populationSizeGA, mutationRateGA, numParentsGA, numberOffspringGA);
            default:
                throw new IllegalArgumentException("Invalid trainer type!");
        }
    }

    // Construct a new network of the requested type
    public static INeuralNetwork buildNewNetwork(NetworkType type) {
        if (type == NetworkType.MultiLayerPerceptron) {
            return new MultiLayerPerceptron(activationFunction, layers);
        } else {
            throw new IllegalArgumentException("Invalid network type!");
        }
    }
}
