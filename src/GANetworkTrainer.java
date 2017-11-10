
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Implementation of the basic Genetic Algorithm.
 * Using uniform N parent crossover and steady state replacement of the worst individuals
 *
 * @author Keely Weisbeck
 */
public class GANetworkTrainer extends NetworkTrainerBase {

    private double mutationRate;
    private int numParents;
    private int numOffspring;

    GANetworkTrainer(int populationSize, double mutationRate, int numParents, int numOffspring) {
        super(populationSize);
        this.mutationRate = mutationRate;
        this.numParents = numParents;
        this.numOffspring = numOffspring;
    }

    @Override
    public INeuralNetwork train(INeuralNetwork network, Dataset samples) {
        startTimer();
        //initialize population
        Population population = new Population();
        for (int i = 0; i < populationSize; i++) {
            population.add(createIndividual(network));
        }

        // Split training set into training and validation sets
        Collections.shuffle(samples);
        Dataset validationSet = new Dataset(samples.subList(0, samples.size() / 10));
        Dataset trainingSet = new Dataset(samples.subList(samples.size() / 10, samples.size()));

        int generation = 0;
        while(shouldContinue(validatePopulation(population, validationSet, generation), generation)) {
            generateOffspring(population);

            // Evaluate all the individuals in the population
            evaluatePopulation(population, trainingSet);
            population.sortByFitness();

            // Remove the weakest individuals to maintain steady state population size
            population = new Population(population.subList(0, populationSize));
            generation++;
        }

        INeuralNetwork bestNetwork = population.getMostFit().buildNetwork();
        printConvergence(NetworkTrainerType.GANetworkTrainer, bestNetwork);
        return bestNetwork;
    }

    /**
     * Perform the reproductive set.
     * 1) Parent Selection, 2) Crossover, 3) Mutation
     */
    private void generateOffspring(Population population) {
        for (int i = 0; i < this.numOffspring; i++) {
            Population parents = selectParents(population, numParents);
            WeightMatrix child = crossover(parents);
            mutation(child.getWeights());
            population.add(child);
        }
    }

    /**
     * Build a new randomly initialized weight matrix for the supplied network.
     */
    protected WeightMatrix createIndividual(INeuralNetwork network) {
        WeightMatrix individual = new WeightMatrix(network);
        List<Double> weights = individual.getWeights();
        // Set weights to random value between [-5.0, 5.0]
        IntStream.range(0, weights.size()).parallel().forEach(i -> weights.set(i, (this.random.nextDouble() * 10) - 5));

        return individual;
    }

    /**
     * Perform uniform crossover between N parents
     */
    protected WeightMatrix crossover(Population parents) {
        int numParents = parents.size();
        List<Double> childWeights = new ArrayList<>(parents.get(0).getWeights());

        // Retrieve the weights and sigmas of the parents from their respective WeightMatrix collections
        List<List<Double>> parentWeights = parents.parallelStream().map(WeightMatrix::getWeights).collect(Collectors.toList());

        // Iterate over the weights, choosing each gene from a random parent in the parent population provided
        IntStream.range(0, childWeights.size()).parallel().forEach(i -> {
            int parentIndex = random.nextInt(numParents);
            childWeights.set(i, parentWeights.get(parentIndex).get(i));
        });

        // Construct and return the child object
        WeightMatrix child = new WeightMatrix(parents.get(0).buildNetwork());
        child.setWeights(childWeights);
        return child;
    }

    /**
     * Iterate over each of the genes in the offspring, probabilistically modifying values according to
     * a gaussian distribution N(0,1)
     */
    private void mutation(List<Double> offspring) {
        IntStream.range(0, offspring.size())
                .filter(i -> random.nextDouble() < mutationRate)
                .parallel()
                .forEach(i -> offspring.set(i, offspring.get(i) + random.nextGaussian()));
    }
}
