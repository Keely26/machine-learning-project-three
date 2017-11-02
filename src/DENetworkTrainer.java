import java.util.List;

public class DENetworkTrainer extends NetworkTrainerBase {

    private int popSize;
    private int numOffspring;
    private double crossOverRate;
    private double beta;

    DENetworkTrainer(int popSize) {
        this.popSize = popSize;
        numOffspring = (int) (1.3 * popSize);
    }

    @Override
    public INeuralNetwork train(INeuralNetwork network, List<Sample> samples) {

        int t = 0;
        //initialize 100 individuals with random weights
        //offspring;
        //fitnesses list
        //while not converge
            // while  numOff < 1.3(pop)
                //randomly select parent1
                // randomly select 3 more disjoint individuals != parent1
                // apply mutation equation to 1,2,3 = parentU
                //cross parentU with parent1
                // add offspring
            // for each individaul evaluate fitness
            // keep only top #pop offspring
            //sort list of offspring low - high
            //create new population from size pop top offspring
        return network;
    }

    public WeightMatrix mutate(WeightMatrix parenta, WeightMatrix parentb, WeightMatrix parentc){
        // a + beta( b - c)

        for(int i = 0; i < a.size(); i ++){

        }
        return a;
    }
    public WeightMatrix crossOver(WeightMatrix parent1, WeightMatrix ParentU){
        //uniform crossover
        return parent1;
    }

    public double fitness(){
        // call calcFitness
        return 0.0;
    }

}
