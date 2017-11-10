
public class NetworkFactory {

    /* MultiLayer Perceptron Parameters */
    private static final int[] layers = new int[]{9, 5, 5, 1};    // Size of each layer
    private static final IActivationFunction activationFunction = new HyperbolicTangent();

    /* Backpropagation Parameters */
    private static final int batchSize = 5;
    private static final double learningRate = 0.2;
    private static final double momentum = 0.0;
    private static final int epochs = 50;

    /* Evolution Strategy Parameters */
    private static final int populationSizeES = 80;
    private static final int numberOffspring = 5;
    private static final int numberParents = 3;
    private static final double mutationRate = 0.01;

    /* Differential Evolution Parameters */
    private static final int populationSizeDE = 100;
    private static final double beta = 0.5;


    /* Genetic Algorithm Parameters */
    private static final int populationSizeGA = 50;
    private static final double mutationRateGA = 0.01;
    private static final int numParentsGA = 2;
    private static final int numberOffspringGA = 20;


    public static INetworkTrainer buildNetworkTrainer(NetworkTrainerType type) {
        switch (type) {
            case BPNetworkTrainer:
                return new BPNetworkTrainer(learningRate, momentum, batchSize, epochs);
            case DENetworkTrainer:
                return new DENetworkTrainer(populationSizeDE, beta);
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
