import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Implementation of Differential Evolution for training a neural net
 *
 * @author Karen Stengel
 */
public class DENetworkTrainer extends NetworkTrainerBase {

    private final double beta;
    private final double crossoverRate;

    private final List<Integer> indexList;

    DENetworkTrainer(int populationSize, double beta, double crossoverRate) {
        super(populationSize);
        this.beta = beta;
        this.crossoverRate = crossoverRate;
        indexList = new ArrayList<>(populationSize);
        for (int i = 0; i < populationSize; i++) {
            indexList.add(i);
        }
    }

    @Override
    public INeuralNetwork train(INeuralNetwork network, Dataset samples) {
        startTimer();
        // Initialize new population
        Population population = IntStream.range(0, populationSize)
                .parallel()
                .mapToObj(i -> initializeIndividual(network))
                .collect(Collectors.toCollection(Population::new));

        // Split training set into training and validation sets
        samples.shuffle();
        Dataset validationSet = new Dataset(samples.subList(0, samples.size() / 10));
        Dataset trainingSet = new Dataset(samples.subList(samples.size() / 10, samples.size()));

        int generation = 0;
        // While stopping conditions haven't been reached, create the next generation and increment
        while (shouldContinue(validatePopulation(population, validationSet, generation), generation, network)) {
            population = createNextGeneration(network, population, trainingSet);
            generation++;
        }

        INeuralNetwork best = bestNetwork.buildNetwork();
        printConvergence(NetworkTrainerType.DENetworkTrainer, best);
        return best;
    }

    /**
     * Generate a new WeightMatrix representation of the given network with random weights
     */
    private WeightMatrix initializeIndividual(INeuralNetwork network) {
        WeightMatrix individual = new WeightMatrix(network);
        List<Double> weights = individual.getWeights();

        // Set weights to random value between [-5.0, 5.0]
        IntStream.range(0, weights.size())
                .parallel()
                .forEach(i -> weights.set(i, (this.random.nextDouble() * 10) - 5));

        return individual;
    }

    /**
     * Loop over the population, generating a child for each individual and adding the stronger of the two to the next
     * generation.
     */
    private Population createNextGeneration(INeuralNetwork network, Population population, Dataset trainingSet) {
        Population nextGeneration = new Population();

        for (int i = 0; i < populationSize; i++) {
            WeightMatrix parent = population.get(i);
            WeightMatrix child = new WeightMatrix(network, createChild(population, i));

            evaluateIndividual(parent, trainingSet);
            evaluateIndividual(child, trainingSet);

            nextGeneration.add(parent.getFitness() < child.getFitness() ? parent : child);
        }
        return nextGeneration;
    }

    /**
     * Given the population and the index of the primary parent, choose three other distinct parents and create a child
     * target vector according to: target = Xa * B (Xb - Xc).
     */
    private List<Double> createChild(Population population, int parentIndex) {
        // Remove parentIndex from list
        Collections.swap(indexList, parentIndex, populationSize - 1);

        // Choose a random index from the list, then swap that index out of range so it cant be chosen twice.
        List<WeightMatrix> auxiliaryParents = new ArrayList<>(3);
        for (int i = 2; i < 5; i++) {
            int randomIndex = random.nextInt(populationSize - i);
            auxiliaryParents.add(population.get(indexList.get(randomIndex)));
            Collections.swap(indexList, randomIndex, populationSize - i);
        }

        List<Double> weightsA = auxiliaryParents.get(0).getWeights();
        List<Double> weightsB = auxiliaryParents.get(1).getWeights();
        List<Double> weightsC = auxiliaryParents.get(2).getWeights();

        // Cross the selected individuals
        List<Double> childWeights = new ArrayList<>();
        for (int i = 0; i < weightsA.size(); i++) {
            if (random.nextDouble() < this.crossoverRate) {
                childWeights.add(weightsA.get(i) + (beta * (weightsB.get(i) - weightsC.get(i))));
            } else {
                childWeights.add(weightsA.get(i));
            }
        }
        return childWeights;
    }
}
