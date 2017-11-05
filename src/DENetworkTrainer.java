//import java.lang.reflect.Array;
import java.util.*;

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
        List<WeightMatrix> population = new ArrayList<WeightMatrix>();
        for (int i = 0; i < popSize; i++){
            //create new weightmatrix using network
            population.add(i, createIndividual(network, new ArrayList<Double>()));
        }

        List<WeightMatrix> offspring = new ArrayList<WeightMatrix>();

        while(t < 5000) {  //while not converge; fix

            // while  numOff < 1.3(pop)
            int numOff = 0;

            while(numOff < numOffspring) {
                //randomly select parent1
                WeightMatrix parent = population.get(new Random().nextInt(popSize));
                // randomly select 3 more disjoint individuals != parent1 or each other
                WeightMatrix parent1 = population.get(new Random().nextInt(popSize));
                while (parent1 == parent){
                    parent1 = population.get(new Random().nextInt(popSize));
                }

                WeightMatrix parent2 = population.get(new Random().nextInt(popSize));
                while (parent2 == parent || parent2 == parent1){
                    parent2 = population.get(new Random().nextInt(popSize));
                }

                WeightMatrix parent3 = population.get(new Random().nextInt(popSize));
                while (parent3 == parent || parent3 == parent1 || parent3 == parent2){
                    parent3 = population.get(new Random().nextInt(popSize));
                }

                // apply mutation equation to 1,2,3 = parentU
                List<Double> parentU = mutate(parent1, parent2, parent3);

                //cross parentU with parent1
                List<Double> childWeights = crossOver(parent, parentU);

                // create new weightmatrix using network; set weights from cross; add to offspring lists
                offspring.add(createIndividual(network, childWeights));


                //create new population from size pop top offspring
                numOff++;

            } // end while

            // for each individual evaluate fitness
            fitness(offspring, samples);
            // keep only top #pop offspring
            while (offspring.size() > popSize) {
                // find min of list and .remove
                int maxIndex = 0;

                for(int i = 0; i < offspring.size(); i++){
                    if(offspring.get(i).getFitness() > offspring.get(maxIndex).getFitness()){
                        maxIndex = i;
                    }
                }
                offspring.remove(maxIndex);
            }
            population = offspring;
            t++;
        } // end while
        // create new network from best
        int minIndex = 0;

        for(int i = 0; i < population.size(); i++) {
            if (population.get(i).getFitness() < population.get(minIndex).getFitness()) {
                minIndex = i;
            }
        }
        return deserializeNetwork(population.get(minIndex));
    }

    public WeightMatrix createIndividual(INeuralNetwork network, List<Double> w){

        WeightMatrix individual = new WeightMatrix(network);

            if(w.isEmpty()) {

                for (int i = 0; i < individual.getWeights().size(); i++) {
                    w.add(i, new Random().nextDouble()*3);//opinions?
                }

                individual.setWeights(w);
            }

            else{
                individual.setWeights(w);
            }

        return individual;
    }

    public List<Double> mutate(WeightMatrix parenta, WeightMatrix parentb, WeightMatrix parentc){
        // u = a + beta( b - c)
        List<Double> a = parenta.getWeights();
        List<Double> b = parentb.getWeights();
        List<Double> c = parentc.getWeights();
        List<Double> u = a;
        for(int i = 0; i < a.size(); i ++){
            u.add(i,a.get(i) + beta * (b.get(i) - c.get(i)));
        }
        return u;
    }

    public List<Double> crossOver(WeightMatrix parent1, List<Double> ParentU){
        //uniform crossover
        List<Double> child = parent1.getWeights();

        for(int i = 0; i <child.size(); i ++) {
            if (!(new Random().nextInt(10) == 0)) {
                // cross @ i
                if (new Random().nextInt() == 1) {
                    child.set(i, ParentU.get(i));
                }
            }
        }
        return child;
    }

    public void fitness(List<WeightMatrix> population, List<Sample> samples) {

        // for each WM in  list create the FFN
        List<INeuralNetwork> FFNPop =  new ArrayList<INeuralNetwork>();
        List<double[]> networkOuts = new ArrayList<double[]>();

        for (int i = 0; i < population.size(); i++) {
            //for each i create a new FFN and save to FFNPop
            FFNPop.add(i, deserializeNetwork(population.get(i)));

            //compute network outputs
            for(int j = 0; j < samples.size(); j++) {
                networkOuts.add(i, execute((FFNPop.get(i)), samples.get(j).inputs));
            }
        }
        // update fitness for each WeightMatrix
        for(int i = 0; i < FFNPop.size(); i++){
            population.get(i).setFitness(meanSquaredError(networkOuts.get(i), samples.get(i).outputs));
        }

    }

}
