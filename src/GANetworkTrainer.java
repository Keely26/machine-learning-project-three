
public class GANetworkTrainer extends NetworkTrainerBase {
    private double crossoverRate;
    private double mutationRate;
    private int rank;

    GANetworkTrainer() {

    }

    @Override
    public INeuralNetwork train(INeuralNetwork network, Dataset samples) {
        //t = 0;
        //initialize population
        //fitness function
        //while not converge
            //select parent --> rank based selection
            //apply crossover
            //apply mutation
            //evaluate fitness of mutation result
            //replace child into population
        // end while

        //return best
        return network;
    }

    private void crossover(WeightMatrix parent1, WeightMatrix parent2) {
    }

    private void mutation(WeightMatrix parent) {
    }

    private void fitnessFunction(Population population) {
        int fitness = 0;

        for(int i = 0; i < population.size(); i++) {

        }
    }
    private void rank() {
    }
}

