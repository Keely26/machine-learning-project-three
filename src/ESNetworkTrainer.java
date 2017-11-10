import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Implementation of (mu, lambda) - Evolution Strategy for neural net training
 *
 * @author Zach Connelly
 */
public class ESNetworkTrainer extends NetworkTrainerBase {

    private final int numParents;
    private final int numOffspring;
    private final double mutationRate;

    ESNetworkTrainer(int populationSize, int numParents, int numOffspring, double mutationRate) {
        super(populationSize);
        this.numParents = numParents;
        this.numOffspring = numOffspring;
        this.mutationRate = mutationRate;
    }

    @Override
    public INeuralNetwork train(INeuralNetwork network, Dataset samples) {
        // Generate initial population
        Population population = IntStream.range(0, populationSize)
                .parallel()
                .mapToObj(i -> createIndividual(network))
                .collect(Collectors.toCollection(Population::new));

        // Split training set into training and validation sets
        Collections.shuffle(samples);
        Dataset validationSet = new Dataset(samples.subList(0, samples.size() / 10));
        Dataset trainingSet = new Dataset(samples.subList(samples.size() / 10, samples.size()));

        for (int i = 0; i < 200; i++) {
            // Perform reproductive step, adding children into population
            generateOffspring(population, i);

            // Remove the least fit individuals to maintain population size
            survivalOfTheFittest(population, trainingSet);

            // Print the performance of population run against the validation set // TODO: Use this is an early cutoff somehow
            validatePopulation(population, validationSet, i);
        }

        // Return the best network
        return population.getMostFit().buildNetwork();
    }

    /**
     * Generate a new WeightMatrix representation of the given network with random weights and sigma values
     */
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

    /**
     * Reproductive step: Select parents, perform crossover and mutation, add children to population
     */
    private void generateOffspring(Population population, int generation) {
        for (int i = 0; i < this.numOffspring; i++) {
            // Select parents
            Population parents = selectParents(population, numParents);
            // Crossover
            WeightMatrix child = crossover(parents);
            // Mutation
            mutate(child, generation % 30 == 0);
            // Add offspring into population
            population.add(child);
        }
    }

    /**
     * Probabilistically mutate using stored probabilities
     */
    private void mutate(WeightMatrix individual, boolean hyperMutate) {
        List<Double> weights = individual.getWeights();
        List<Double> sigmas = individual.getSigmas();

        double rate = hyperMutate ? this.mutationRate * 10 : this.mutationRate;

        IntStream.range(0, weights.size() / 2)
                .filter(i -> this.random.nextFloat() < rate)
                .parallel()
                .forEach(i -> weights.set(i, weights.get(i) + this.random.nextGaussian() * sigmas.get(i) * (hyperMutate ? 3 : 1)));
    }

    /**
     * Perform uniform crossover between N parents
     */
    private WeightMatrix crossover(Population parents) {
        List<Double> childWeights = new ArrayList<>(parents.get(0).getWeights());
        List<Double> childSigmas = new ArrayList<>(parents.get(0).getWeights());

        // Retrieve the weights and sigmas of the parents from their respective WeightMatrix collections
        List<List<Double>> parentWeights = parents.parallelStream().map(WeightMatrix::getWeights).collect(Collectors.toList());
        List<List<Double>> parentSigmas = parents.parallelStream().map(WeightMatrix::getSigmas).collect(Collectors.toList());

        // Iterate over the weights, choosing each gene from a random parent in the parent population provided
        IntStream.range(0, childWeights.size()).parallel().forEach(i -> {
            int parentIndex = random.nextInt(numParents);
            childWeights.set(i, parentWeights.get(parentIndex).get(i));
        });

        // Iterate over the weights, choosing each gene from a random parent in the parent population provided
        IntStream.range(0, childSigmas.size()).parallel().forEach(i -> {
            int parentIndex = random.nextInt(numParents);
            childSigmas.set(i, parentSigmas.get(parentIndex).get(i));
        });

        // Construct and return the child object
        WeightMatrix child = new WeightMatrix(parents.get(0).buildNetwork());
        child.setWeights(childWeights);
        child.setSigmas(childSigmas);
        return child;
    }

    /**
     * Run each individual against the training set to evaluate its fitness
     * then remove the least fit individuals from the population to maintain size
     */
    private void survivalOfTheFittest(Population population, Dataset trainingData) {
        evaluatePopulation(population, trainingData);

        // Remove least fit individuals
        population.sortByFitness();
        while (population.size() > populationSize) {
            population.remove(population.size() - 1);
        }

        // Adjust sigmas
        for (int i = 0; i < populationSize; i++) {
            population.get(i).adjustSigma(i > populationSize / 5);
        }
    }
}
