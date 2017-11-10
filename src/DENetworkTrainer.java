import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class DENetworkTrainer extends NetworkTrainerBase {

    private double beta;
    private final List<Integer> indexList;

    DENetworkTrainer(int populationSize, double beta) {
        super(populationSize);
        this.beta = beta;
        indexList = new ArrayList<>(populationSize);
        for (int i = 0; i < populationSize; i++) {
            indexList.add(i);
        }
    }

    @Override
    public INeuralNetwork train(INeuralNetwork network, Dataset samples) {
        double startTime = System.nanoTime();

        // Initialize new population
        Population population = IntStream.range(0, populationSize)
                .parallel()
                .mapToObj(i -> initializeIndividual(network))
                .collect(Collectors.toCollection(Population::new));

        // Split training set into training and validation sets
        Collections.shuffle(samples);
        Dataset validationSet = new Dataset(samples.subList(0, samples.size() / 10));
        Dataset trainingSet = new Dataset(samples.subList(samples.size() / 10, samples.size()));

        int generation = 0;
        while (shouldContinue(validatePopulation(population, validationSet, generation))) {
            population = createNextGeneration(network, population, trainingSet);
            generation++;
        }

        System.out.println("DE convergence time: " + (System.nanoTime() - startTime) / 1000000000.0 + " seconds.");
        return population.getMostFit().buildNetwork();
    }

    /**
     * Generate a new WeightMatrix representation of the given network with random weights
     */
    private WeightMatrix initializeIndividual(INeuralNetwork network) {
        WeightMatrix individual = new WeightMatrix(network);
        List<Double> weights = individual.getWeights();

        // Set weights to random value between [-5.0, 5.0]
        IntStream.range(0, weights.size()).parallel().forEach(i -> weights.set(i, (this.random.nextDouble() * 10) - 5));

        return individual;
    }

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

    private List<Double> createChild(Population population, int parentIndex) {
        // Remove parentIndex from list
        Collections.swap(indexList, parentIndex, populationSize - 1);

        List<WeightMatrix> auxiliaryParents = new ArrayList<>(3);
        for (int i = 2; i < 5; i++) {
            int randomIndex = random.nextInt(populationSize - i);
            auxiliaryParents.add(population.get(indexList.get(randomIndex)));
            Collections.swap(indexList, randomIndex, populationSize - i);
        }

        List<Double> weightsA = auxiliaryParents.get(0).getWeights();
        List<Double> weightsB = auxiliaryParents.get(1).getWeights();
        List<Double> weightsC = auxiliaryParents.get(2).getWeights();

        return IntStream.range(0, weightsA.size())
                .parallel()
                .mapToObj(i -> weightsA.get(i) + (beta * (weightsB.get(i) - weightsC.get(i))))
                .collect(Collectors.toList());
    }
}
