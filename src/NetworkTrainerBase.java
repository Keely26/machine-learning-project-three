import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

public class NetworkTrainerBase implements INetworkTrainer {

    protected final Random random = new Random(System.nanoTime());
    protected final int populationSize;

    protected int cutoffCounter = 0;
    protected double runningAvg = 0.0;
    protected double startTime = 0.0;

    protected int bestGeneration = 0;
    protected double bestError = Double.MAX_VALUE;
    protected WeightMatrix bestNetwork;

    NetworkTrainerBase(int populationSize) {
        this.populationSize = populationSize;
    }

    @Override
    public INeuralNetwork train(INeuralNetwork network, Dataset samples) {
        System.out.println("Train should be called an instance of the base, not the base class itself!!");
        System.exit(-1);
        return null;
    }

    /**
     * Select N individuals from the population without duplicates using rank based selection according to an
     * exponential distribution
     */
    protected Population selectParents(Population population, int numParents) {
        List<Integer> parentIndices = new ArrayList<>(numParents);
        while (parentIndices.size() < numParents) {
            // Select parent indices according to an exponential distribution // TODO: Find cleaner way to do this
            int temp = (int) (StrictMath.log(1 - random.nextDouble()) * -populationSize) % populationSize;
            if (!parentIndices.contains(temp)) {
                parentIndices.add(temp);
            }
        }
        // Pull parents out of population based on selected indices
        Population parents = new Population();
        for (int i = 0; i < numParents; i++) {
            parents.add(population.get(parentIndices.get(i)));
        }
        return parents;
    }

    /**
     * Perform a multi threaded evaluation of each individual in the population by calling evaluate individual foreach.
     */
    protected void evaluatePopulation(Population population, Dataset trainingData) {
        population.parallelStream().forEach(individual -> NetworkTrainerBase.this.evaluateIndividual(individual, trainingData));
    }

    /**
     * Evaluate the performance of a single individual on a provided dataset.
     * Using the provided individual, map over the elements of the training set and sum the output errors.
     * Set the fitness of the individual.
     */
    protected void evaluateIndividual(WeightMatrix individual, Dataset trainingSet) {
        individual.setFitness(trainingSet
                .parallelStream()
                .mapToDouble(sample -> meanSquaredError(individual.buildNetwork().execute(sample.inputs), sample.outputs))
                .sum());
    }

    /**
     * Evaluate a population according to the provided validation set. Print the error to stdout.
     */
    protected double validatePopulation(Population population, Dataset validationSet, int generation) {
        // Calculate the total validation error of the population
        double error = population
                .parallelStream()
                .mapToDouble(individual -> validationSet
                        .stream()
                        .mapToDouble(sample -> meanSquaredError(individual.buildNetwork().execute(sample.inputs), sample.outputs))
                        .sum() / validationSet.size())
                .sum();
        // Print to stdout
        System.out.println("Generation: " + generation + "\t\t" + "Validation set error: " + error / validationSet.size());
        return error;
    }

    /**
     * Given the expected and actual outputs, compute the normalized mean squared error of their difference.
     */
    protected double meanSquaredError(double[] networkOutputs, double[] expectedOutputs) {
        assert networkOutputs.length == expectedOutputs.length;

        // Calculate the sum over the squared error for each output value
        double errorSum = IntStream.range(0, networkOutputs.length)
                .parallel()
                .mapToDouble(i -> Math.pow(networkOutputs[i] - expectedOutputs[i], 2))
                .sum();

        // Normalize and return error
        return errorSum / (networkOutputs.length * expectedOutputs.length);
    }

    /**
     * Record the starting time and reset counters.
     */
    protected void startTimer() {
        this.startTime = System.nanoTime();
        this.cutoffCounter = 0;
        this.runningAvg = 9999;
    }

    /**
     * Terminate if the average change in performance over the last 5 generations is less than 0.5%
     */
    protected boolean shouldContinue(double validationError, int generation, INeuralNetwork network) {
        if (validationError < bestError) {
            bestGeneration = generation;
            bestError = validationError;
            bestNetwork = network.constructWeightMatrix();
        }
        runningAvg = ((runningAvg * 9) + validationError) / 10;
        if ((runningAvg - validationError) / validationError < 0.005) {
            cutoffCounter++;
        } else {
            cutoffCounter = 0;
        }
        return generation < 5000 && cutoffCounter < 20;
    }

    /**
     * Calculates the convergence time of the trainer, setting this value on the network and
     * printing convergence time to the stdout.
     */
    protected void printConvergence(NetworkTrainerType type, INeuralNetwork network) {
        double convergenceTime = (System.nanoTime() - startTime) / 1000000000.0;
        network.setConvergence(convergenceTime);
        StringBuilder output = new StringBuilder();
        output.append("\n");
        switch (type) {
            case DENetworkTrainer:
                output.append("Differential Evolution Convergence Time: ");
                break;
            case GANetworkTrainer:
                output.append("Genetic Algorithm Convergence Time: ");
                break;
            case ESNetworkTrainer:
                output.append("Evolution Strategy Convergence Time: ");
                break;
            case BPNetworkTrainer:
                output.append("Backpropagation Convergence Time: ");
                break;
        }
        output.append(convergenceTime);
        output.append(" seconds.");
        System.out.println(output.toString());
    }
}
