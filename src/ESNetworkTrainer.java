import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Implementation of (mu, lambda) - Evolution Strategy for neural net training
 *
 * @author Zach Connelly
 */
public class ESNetworkTrainer extends NetworkTrainerBase {

    private final int populationSize;
    private final int numParents;
    private final int numOffspring;
    private final double mutationRate;

    private final Random random = new Random(System.nanoTime());

    ESNetworkTrainer(int populationSize, int numParents, int numOffspring, double mutationRate) {
        this.populationSize = populationSize;
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

        for (int i = 0; i < 50000; i++) {
            // Perform reproductive step, adding children into population
            generateOffspring(population, i);

            // Remove the least fit individuals to maintain population size
            survivalOfTheFittest(population, trainingSet);

            // Print the performance of population run against the validation set // TODO: Use this is an early cutoff somehow
            validateFitness(population, validationSet, i);
        }

        // Return the best network
        return deserializeNetwork(population.getMostFit());
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
        for (int j = 0; j < this.numOffspring; j++) {
            // Select parents
            Population parents = getParents(population);
            // Crossover
            WeightMatrix child = crossover(parents);
            // Mutation
            if (generation % 50 == 0) {
                hyperMutate(child);
            } else {
                mutate(child);
            }
            // Add offspring into population
            population.add(child);
        }
    }

    /**
     * Select N individuals from the population without duplicates using rank based selection according to an
     * exponential distribution
     */
    private Population getParents(Population population) {
        population.sortByFitness();

        List<Integer> parentIndices = new ArrayList<>(this.numParents);
        while (parentIndices.size() < this.numParents) {
            // Select parent indices according to an exponential distribution // TODO: Find cleaner way to do this
            int temp = (int) (StrictMath.log(1 - random.nextDouble()) * -populationSize) % populationSize;
            if (!parentIndices.contains(temp)) {
                parentIndices.add(temp);
            }
        }
        // Pull parents out of population based on selected indices
        Population parents = new Population();
        for (int i = 0; i < this.numParents; i++) {
            parents.add(population.get(parentIndices.get(i)));
        }
        return parents;
    }

    /**
     * Probabilistically mutate using stored probabilities
     */
    private void mutate(WeightMatrix individual) {
        List<Double> weights = individual.getWeights();
        List<Double> sigmas = individual.getSigmas();

        IntStream.range(0, weights.size() / 2)
                .filter(i -> this.random.nextFloat() < this.mutationRate)
                .parallel()
                .forEach(i -> weights.set(i, weights.get(i) + this.random.nextGaussian() * sigmas.get(i)));
    }

    // TODO: Consolidate these two methods using Consumer or boolean flag

    /**
     * Perform a more extreme mutation
     * Uses five times the mutation rate and a large value change
     */
    private void hyperMutate(WeightMatrix individual) {
        List<Double> weights = individual.getWeights();
        List<Double> sigmas = individual.getSigmas();

        IntStream.range(0, weights.size() / 2)
                .filter(i -> this.random.nextFloat() < this.mutationRate / 5)
                .parallel()
                .forEach(i -> weights.set(i, weights.get(i) + this.random.nextGaussian() * sigmas.get(i) * 3));
    }

    /**
     * Perform uniform crossover between N parents
     */
    private WeightMatrix crossover(Population parents) {
        List<Double> childWeights = new ArrayList<>(parents.get(0).getWeights());
        List<Double> childSigmas = new ArrayList<>(parents.get(0).getWeights());

        List<List<Double>> parentWeights = parents.stream().parallel().map(WeightMatrix::getWeights).collect(Collectors.toList());
        List<List<Double>> parentSigmas = parents.stream().parallel().map(WeightMatrix::getSigmas).collect(Collectors.toList());

        IntStream.range(0, childWeights.size()).parallel().forEach(i -> {
            int parentIndex = random.nextInt(numParents);
            childWeights.set(i, parentWeights.get(parentIndex).get(i));
        });

        IntStream.range(0, childSigmas.size()).parallel().forEach(i -> {
            int parentIndex = random.nextInt(numParents);
            childSigmas.set(i, parentSigmas.get(parentIndex).get(i));
        });

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
        evaluateFitness(population, trainingData);

        // Remove least fit individuals
        population.sortByFitness();
        while (population.size() > populationSize) {
            population.remove(population.size() - 1);
        }

        // Adjust sigmas
        for (int i = 0; i < populationSize; i++) {
            if (i < populationSize / 5) {
                population.get(i).decreaseSigma();
            } else {
                population.get(i).increaseSigma();
            }
        }
    }
}
