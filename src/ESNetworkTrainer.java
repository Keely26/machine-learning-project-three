import java.util.List;

public class ESNetworkTrainer extends NetworkTrainerBase {

    private final int populationSize;
    private final int numOffspring;

    ESNetworkTrainer(int populationSize, int numOffspring) {
        this.populationSize = populationSize;
        this.numOffspring = numOffspring;
    }

    @Override
    public INeuralNetwork train(INeuralNetwork network, List<Sample> samples) {
        // Generate initial population randomly
        // Do
            // Cross
            // Mutate
            // Evaluate
            // Choose new population
        // While some condition

        return network;
    }
}
