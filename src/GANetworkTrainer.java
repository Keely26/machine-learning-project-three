import java.util.List;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class GANetworkTrainer extends NetworkTrainerBase {

    private double mutationRate;
    private int numParents;

    GANetworkTrainer(int populationSize, double mutationRate, int numParents) {
        super(populationSize);
        this.mutationRate = mutationRate;
        this.numParents = numParents;
    }

    @Override
    public INeuralNetwork train(INeuralNetwork network, Dataset samples) {

        //initialize population
        Population population = new Population();
        for (int i = 0; i < populationSize; i++) {
            population.add(createIndividual(network));
        }

        // Split training set into training and validation sets
        Collections.shuffle(samples);
        Dataset validationSet = new Dataset(samples.subList(0, samples.size() / 10));
        Dataset trainingSet = new Dataset(samples.subList(samples.size() / 10, samples.size()));

        //while not converge
        for (int i = 0; i < 500; i++) {
            //fitness function
            evaluatePopulation(population, trainingSet);

            for (int j = 0; j < populationSize / 2; j++) {
                //select parents
                Population parents = selectParents(population, numParents);
                //apply crossover
                List<Double> childWeights = crossover(parents);
                //apply mutation
                mutation(childWeights, mutationRate);
                //evaluate fitness of mutation result
                WeightMatrix child = new WeightMatrix(network, childWeights);
                population.add(child);
            }
            population = new Population(population.subList(0, populationSize));

            validatePopulation(population, validationSet, i);
        }
        return network;
    }

    private WeightMatrix createIndividual(INeuralNetwork network) {
        WeightMatrix individual = new WeightMatrix(network);
        List<Double> weights = individual.getWeights();
        // Set weights to random value between [-5.0, 5.0]
        IntStream.range(0, weights.size()).parallel().forEach(i -> weights.set(i, (this.random.nextDouble() * 10) - 5));

        // Set sigmas to random value between [-2.0, 2.0]
        List<Double> sigmas = IntStream.range(0, weights.size()).parallel().mapToObj(i -> (this.random.nextDouble() * 4) - 2).collect(Collectors.toList());
        individual.setSigmas(sigmas);

        return individual;
    }


    private List<Double> crossover(Population parents) {
        List<Double> child = new ArrayList<>();
        int size = parents.get(0).getWeights().size();
        for (int i = 0; i < size; i++) {
            child.add(parents.get(random.nextInt(parents.size())).getWeights().get(i));
        }
        return child;
    }


    private void mutation(List<Double> offspring, double mutationRate) {
        for (int i = 0; i < offspring.size(); i++) {
            double gene = offspring.get(i);
            if (random.nextDouble() <= mutationRate) {
                offspring.set(i, gene + random.nextGaussian());
            }
        }
    }

    private void rank(Population population) {
        //sort the population by fitness
        population.sortByFitness();
    }
}
