import java.util.List;

public class DENetworkTrainer extends NetworkTrainerBase {

    private double crossOverRate;
    private double beta;

    DENetworkTrainer() {

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



        return network;
    }

    public WeightMatrix mutate(WeightMatrix a, WeightMatrix b, WeightMatrix c){
        // a + beta( b - c)
        return a;
    }
    public WeightMatrix crossOver(WeightMatrix parent1, WeightMatrix ParentU){
        return parent1;
    }

    public void fitness(){
        s
    }

}
