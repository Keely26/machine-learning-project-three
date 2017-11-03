import java.util.List;
import java. util.Random;

public class ESNetworkTrainer extends NetworkTrainerBase {

    private final int populationSize;
    private final int numOffspring;
    private double crossOverRate;

    ESNetworkTrainer(int populationSize, int numOffspring) {
        this.populationSize = populationSize;
        this.numOffspring = numOffspring;
    }

    @Override
    public INeuralNetwork train(INeuralNetwork network, List<Sample> samples) {
        int t = 0;
        // Generate initial population randomly with random SDs
        // Do
            // Cross
            // Mutate
            // Evaluate
            // Choose new population
        // While some condition

        return network;
    }

    public WeightMatrix mutate(WeightMatrix individual){
        // probabilistically mutate using stored probabilities
        for( int i = 0; i < 10; i++) {

            if(new Random().nextInt(1000) == 0){
                // mutate individual[i]
            }
        }
        return individual;
    }

    public List<Double> crossOver( WeightMatrix parent1, WeightMatrix[] parents){
        // uniform cross over
        List<Double> child = parent1.getWeights();
        for( int i = 0; i < parent1.getWeights().size(); i++) {

            if(!(new Random().nextInt(10) == 0)){
                // cross @ i
            }
        }
        return child;
    }

    public double fitness(WeightMatrix individual){
        // call calcFitness
        return 0.0;
    }
}
