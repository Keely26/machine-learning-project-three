import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

public class NetworkTrainerBase implements INetworkTrainer {

    protected final Random random = new Random(System.nanoTime());
    protected final int populationSize;

    protected int cutoffCounter = 0;
    protected double runningAvg = 0.0;
    private double startTime;

    NetworkTrainerBase(int populationSize) {
        this.populationSize = populationSize;
    }


    @Override
    public INeuralNetwork train(INeuralNetwork network, Dataset samples) {
        System.out.println("Train should be called an instance of the base, not the base class itself!!");
        System.exit(-1);
        return null;
    }

    // Take a network and a set of inputs, run them through the network and return the outputs
    protected double[] execute(INeuralNetwork network, double[] inputs) {
        return network.execute(inputs);
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

    protected void evaluatePopulation(Population population, Dataset trainingData) {
        population.parallelStream().forEach((WeightMatrix individual) -> {
            //System.out.println(individual);
            evaluateIndividual(individual, trainingData);
        });
    }

    protected void evaluateIndividual(WeightMatrix individual, Dataset trainingData) {
        double fitness = trainingData
                .parallelStream()
                .mapToDouble((Sample sample) -> {
                    INeuralNetwork network = individual.buildNetwork();
                    double[] networkOutputs = network.execute(sample.inputs);
                    return meanSquaredError(networkOutputs, sample.outputs);
                })
                .sum();
        individual.setFitness(fitness);
    }

    protected double validatePopulation(Population population, Dataset validationSet, int generation) {
        double error = population
                .parallelStream()
                .mapToDouble(individual -> validationSet.stream()
                        .mapToDouble(sample -> meanSquaredError(individual.buildNetwork().execute(sample.inputs), sample.outputs)).sum() / validationSet.size())
                .sum();
        System.out.println("Generation: " + generation + "\t\t" + "Validation set error: " + error / validationSet.size());
        return error;
    }

    // Compute the normalized squared error between a set of outputs and their true values
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

    protected void startTimer() {
        this.startTime = System.nanoTime();
    }

    /**
     * Terminate if the average change in performance over the last 5 generations is less than 0.5%
     */
    protected boolean shouldContinue(double validationError, int generation) {
        runningAvg = ((runningAvg * 4) + validationError) / 5;
        if (Math.abs(validationError - runningAvg) / validationError < 0.01) {
            cutoffCounter++;
        } else {
            cutoffCounter = 0;
        }
        return generation < 5000 && cutoffCounter < 7;
    }

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
