import java.util.List;

public class GANetworkTrainer extends NetworkTrainerBase {

    GANetworkTrainer() {

    }

    @Override
    public INeuralNetwork train(INeuralNetwork network, List<Sample> samples) {
        //t = 0;
        //initialize population
        //fitness function
        //while not converge
            // select parent
            //apply cross over
            //apply mutation
            //evaluate fitness of mutation result
            //replace child into population
        // end while

        //return best
        return network;
    }

    private void crossover(parent1, parent2) {
    }

    private void mutation(parent) {
    }

    private void fitnessFunction(population[]) {
        int fitness = 0;

        for(int i = 0; i < population.size(); i++) {

        }
    }
}
