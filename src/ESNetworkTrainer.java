import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

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
        Population population = new Population();
        for (int i = 0; i < populationSize; i++) {
            population.add(createIndividual(network));
        }

        // Split training set into training and validation sets
        Collections.shuffle(samples);
        Dataset validationSet = new Dataset(samples.subList(0, samples.size() / 10));
        Dataset trainingSet = new Dataset(samples.subList(samples.size() / 10, samples.size()));

        for (int i = 0; i < 500; i++) {
            // Perform reproductive step, adding children into population
            generateOffspring(network, population);

            // Remove the least fit individuals to maintain population size
            survivalOfTheFittest(population, trainingSet);

            validate(population.get(0), validationSet);
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

        for (int i = 0; i < weights.size(); i++) {
            if (i < weights.size() / 2) {
                weights.set(i, (this.random.nextDouble() * 10) - 5);    // Set weights to random value between [-5.0, 5.0]
            } else {
                weights.set(i, (this.random.nextDouble() * 4) - 2);     // Set sigmas to random value between [-2.0, 2.0]
            }
        }

        return individual;
    }

    /**
     * Reproductive step: Select parents, perform crossover and mutation, add children to population
     */
    private void generateOffspring(INeuralNetwork network, Population population) {
        for (int j = 0; j < this.numOffspring; j++) {
            List<WeightMatrix> parents = getParents(population);
            List<Double> childWeights = crossover(parents);
            List<Double> mutatedChildWeights = mutate(childWeights);
            WeightMatrix child = new WeightMatrix(network, mutatedChildWeights);
            population.add(child);
        }
    }

    /**
     * Select N random individuals from the population without duplicates
     */
    private List<WeightMatrix> getParents(Population population) {
        List<WeightMatrix> parents = new ArrayList<>(population);
        Collections.shuffle(parents);
        return parents.subList(0, this.numParents);
    }

    /**
     * Probabilistically mutate using stored probabilities
     */
    private List<Double> mutate(List<Double> individual) {
        List<Double> weights = individual.subList(0, individual.size() / 2);
        List<Double> sigmas = individual.subList(individual.size() / 2, individual.size());

        for (int i = 0; i < weights.size() / 2; i++) {
            if (this.random.nextFloat() < this.mutationRate) {
                weights.set(i, weights.get(i) + this.random.nextGaussian() * sigmas.get(i));
            }
        }

        return individual;
    }

    /**
     * Perform uniform crossover between N parents
     */
    private List<Double> crossover(List<WeightMatrix> parents) {
        List<Double> childWeights = new ArrayList<>(parents.get(0).getWeights());

        List<List<Double>> parentWeights = parents.stream().map(WeightMatrix::getWeights).collect(Collectors.toList());
        for (int i = 0; i < childWeights.size(); i++) {
            int parentIndex = random.nextInt(numParents);
            childWeights.set(i, parentWeights.get(parentIndex).get(i));
        }

        return childWeights;
    }

    /**
     * Remove the least fit individuals from the population to maintain size
     */
    private void survivalOfTheFittest(Population population, Dataset trainingData) {
        evaluateFitness(population, trainingData);

        // Remove least fit individuals
        population.sortByFitness();
        while (population.size() > populationSize) {
            population.remove(population.size() - 1);
        }
    }

    /**
     *  Use the validation set to determine the quality of the best individual
     */
    private void validate(WeightMatrix mostFit, Dataset validationSet) {
        INeuralNetwork network = mostFit.buildNetwork();
        double error = 0.0;
        for (Sample sample : validationSet) {
            error += meanSquaredError(network.execute(sample.inputs), sample.outputs);
        }
        System.out.println(error / validationSet.size());
    }
}
