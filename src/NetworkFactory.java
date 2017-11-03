
public class NetworkFactory {

    /* Tunable Parameters */
    private static final int[] layers = new int[]{2, 3, 2};    // Size of each layer
    private static final int batchSize = 5;
    private static final double learningRate = 0.05;
    private static final double momentum = 0;
    private static final IActivationFunction activationFunction = new HyperbolicTangent();
    private static final int epochs = 5000;

    public static INetworkTrainer buildNetworkTrainer(NetworkTrainerType type) {
        switch (type) {
            case BPNetworkTrainer:
                return new BPNetworkTrainer(learningRate, momentum, batchSize, epochs);
            case DENetworkTrainer:
                return new DENetworkTrainer(100);
            case ESNetworkTrainer:
                return new ESNetworkTrainer(100, 10);
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
