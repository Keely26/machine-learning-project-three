import java.util.ArrayList;
import java.util.List;
import java. util.Random;

public class ESNetworkTrainer extends NetworkTrainerBase {

    private final int populationSize;
    private final int numOffspring;
    private double crossOverRate;
    private int parentNum = 2;

    ESNetworkTrainer(int populationSize, int numOffspring) {
        this.populationSize = populationSize;
        this.numOffspring = numOffspring;
    }

    @Override
    public INeuralNetwork train(INeuralNetwork network, List<Sample> samples) {
        int t = 0;
        // Generate initial population randomly with random SDs
        List<WeightMatrix> population = new ArrayList<WeightMatrix>();

        for (int i = 0; i < populationSize; i++){
            //create new weightmatrix using network
            population.add(i, createIndividual(network, new ArrayList<Double>()));
        }
        // Do
        while(t < 5000) { //while not converge; fix
            int numOff = 0;
            while(numOff < numOffspring) {
                // randomly select parent1
                WeightMatrix parent1 = population.get(new Random().nextInt(populationSize));

                //randomly select parentNum parents and add to list; make sure that
                // they are all different
                WeightMatrix[] parents = new WeightMatrix[parentNum];

                for(int i = 0; i < parentNum; i++){ //fix this
                    // randomly select while < parentNum
                    WeightMatrix parentTemp = population.get(new Random().nextInt(populationSize));
                    while(parentTemp == parent1) {
                        parentTemp = population.get(new Random().nextInt(populationSize));

                        for(int j = 0; j < i; j++){

                            if(parentTemp == parents[i]){
                                parentTemp = population.get(new Random().nextInt(populationSize));
                            }
                        }
                    }
                    parents[i] = parentTemp;
                    //enforce that none are the same
                }
                // Cross
                List<Double> child = crossOver(parent1, parents);
                // Mutate
                List<Double> mutChild = mutate(child);
                //add to population
                population.add(createIndividual(network, mutChild));

                numOff++;
            }// end while

            // Evaluate
            fitness(population, samples);
            // Choose new population; remove worst individuals
            while (population.size() > populationSize) {
                // find min of list and .remove
                int minIndex = 0;

                for(int i = 0; i < population.size(); i++){
                    // if fitness at i is worse (error is larger)
                    if(population.get(i).getFitness() > population.get(minIndex).getFitness()){
                        minIndex = i;
                    }
                }
                population.remove(minIndex);
            }
            t++;
        }//end while

        // create new network from best
        int minIndex = 0;

        for(int i = 0; i < population.size(); i++) {
            if (population.get(i).getFitness() < population.get(minIndex).getFitness()) {
                minIndex = i;
            }
        }
        return deserializeNetwork(population.get(minIndex));
    }

    public WeightMatrix createIndividual(INeuralNetwork network, List<Double> w){ //fix

        WeightMatrix individual = new WeightMatrix(network);

        if(w.isEmpty()) {

            for (int i = 0; i < (individual.getWeights().size()/2); i++) {
                w.add(i, new Random().nextDouble()*3); //opinions?
            }
            for (int i = (individual.getWeights().size()/2 + 1); i < (individual.getWeights().size()); i++) {
                w.add(i, new Random().nextDouble());
            }
            individual.setWeights(w);
        }

        else{
            individual.setWeights(w);
        }

        return individual;
    }

    public List<Double> mutate(List<Double> individual){
        // probabilistically mutate using stored probabilities

        int dimension =(individual.size()/2);

        for( int i = 0; i < dimension; i++) {

            int sigmaIndex = i + dimension;

            if(new Random().nextInt(1000) == 0) { // mutation rate == 0.001
                // mutate individual[i]
                individual.set(i, (individual.get(i) + (new Random().nextGaussian()*individual.get(sigmaIndex))));
            }
        }
        // sigma mutation?
        return individual;
    }

    public List<Double> crossOver( WeightMatrix parent1, WeightMatrix[] parents){
        // uniform cross over of weights and sigmas
        List<Double> child = parent1.getWeights();

        for( int i = 0; i < child.size(); i++) {

            if(!(new Random().nextInt(10) == 0)){
                // cross @ i
                int parentChose = new Random().nextInt(parents.length);
                child.set(i, parents[parentChose].getWeights().get(i));
            }
        }
        return child;
    }

    public void fitness(List<WeightMatrix> population, List<Sample> samples){
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

