import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Tester class provides a means to initialize and train neural networks
 * Additionally a five by two cross fold validation function is provided to test and compare networks
 */
public class Tester {

    private static final DatasetType datasetType = DatasetType.Wine;

    public static void main(String[] args) {
        setFileOut();
        Dataset dataset = DatasetFactory.buildDataSet(datasetType);
        //testOne(dataset, NetworkFactory.buildNetworkTrainer(NetworkTrainerType.GANetworkTrainer));
        testAll(dataset);
    }

    private static void testOne(Dataset dataset, INetworkTrainer trainer) {
        trainer.train(NetworkFactory.buildNewNetwork(NetworkType.MultiLayerPerceptron), dataset);
    }

    private static void testAll(Dataset dataset) {
        List<INetworkTrainer> trainers = new ArrayList<>();
        trainers.add(NetworkFactory.buildNetworkTrainer(NetworkTrainerType.GANetworkTrainer));
        trainers.add(NetworkFactory.buildNetworkTrainer(NetworkTrainerType.ESNetworkTrainer));
        trainers.add(NetworkFactory.buildNetworkTrainer(NetworkTrainerType.DENetworkTrainer));
        trainers.add(NetworkFactory.buildNetworkTrainer(NetworkTrainerType.BPNetworkTrainer));
        crossValidate(trainers, dataset);
    }


    /**
     * Execute a 5x2 cross validation comparing each of the trainers provided
     */
    public static void crossValidate(List<INetworkTrainer> trainers, Dataset dataset) {
        List<Double> ESErrors = new ArrayList<>();
        List<Double> DEErrors = new ArrayList<>();
        List<Double> GAErrors = new ArrayList<>();
        List<Double> BPErrors = new ArrayList<>();

        for (int k = 0; k < 5; k++) {
            Collections.shuffle(dataset);
            Dataset testSet = dataset.getTestingSet();
            Dataset trainSet = dataset.getTrainingSet();

            GAErrors.addAll(computeFold(trainSet, testSet, NetworkFactory.buildNewNetwork(NetworkType.MultiLayerPerceptron), trainers.get(0)));
            logStep(k, NetworkTrainerType.GANetworkTrainer);
            ESErrors.addAll(computeFold(trainSet, testSet, NetworkFactory.buildNewNetwork(NetworkType.MultiLayerPerceptron), trainers.get(1)));
            logStep(k, NetworkTrainerType.ESNetworkTrainer);
            DEErrors.addAll(computeFold(trainSet, testSet, NetworkFactory.buildNewNetwork(NetworkType.MultiLayerPerceptron), trainers.get(2)));
            logStep(k, NetworkTrainerType.DENetworkTrainer);
            BPErrors.addAll(computeFold(trainSet, testSet, NetworkFactory.buildNewNetwork(NetworkType.MultiLayerPerceptron), trainers.get(3)));
            logStep(k, NetworkTrainerType.BPNetworkTrainer);

            //training set and test set are swapped
            GAErrors.addAll(computeFold(testSet, trainSet, NetworkFactory.buildNewNetwork(NetworkType.MultiLayerPerceptron), trainers.get(0)));
            logStep(k, NetworkTrainerType.GANetworkTrainer);
            ESErrors.addAll(computeFold(testSet, trainSet, NetworkFactory.buildNewNetwork(NetworkType.MultiLayerPerceptron), trainers.get(1)));
            logStep(k, NetworkTrainerType.ESNetworkTrainer);
            DEErrors.addAll(computeFold(testSet, trainSet, NetworkFactory.buildNewNetwork(NetworkType.MultiLayerPerceptron), trainers.get(2)));
            logStep(k, NetworkTrainerType.DENetworkTrainer);
            BPErrors.addAll(computeFold(testSet, trainSet, NetworkFactory.buildNewNetwork(NetworkType.MultiLayerPerceptron), trainers.get(3)));
            logStep(k, NetworkTrainerType.BPNetworkTrainer);
        }

        double mean = calcMean(GAErrors);
        double SD = calcStandardDeviation(mean, GAErrors);
        printStats(mean, SD, NetworkTrainerType.GANetworkTrainer);

        mean = calcMean(ESErrors);
        SD = calcStandardDeviation(mean, ESErrors);
        printStats(mean, SD, NetworkTrainerType.ESNetworkTrainer);

        mean = calcMean(DEErrors);
        SD = calcStandardDeviation(mean, DEErrors);
        printStats(mean, SD, NetworkTrainerType.DENetworkTrainer);

        mean = calcMean(BPErrors);
        SD = calcStandardDeviation(mean, BPErrors);
        printStats(mean, SD, NetworkTrainerType.BPNetworkTrainer);
    }

    private static List<Double> computeFold(Dataset trainSet, Dataset testSet, INeuralNetwork network, INetworkTrainer trainer) {
        trainer.train(network, trainSet);
        return getApproximationErrors(testSet, network);
    }

    // Iterates through testing set and calculates the approximated values and the error of the samples of the supplied network
    private static List<Double> getApproximationErrors(List<Sample> testSet, INeuralNetwork network) {
        List<Double> totalError = new ArrayList<>(testSet.size());
        // Get the network's approximation
        // Add the error for each sample to the total error
        testSet.parallelStream()
                .forEach(sample -> {
                    double[] networkOutput = network.execute(sample.inputs);
                    totalError.add(IntStream
                            .range(0, networkOutput.length)
                            .mapToDouble(j -> Math.abs(networkOutput[j] - sample.outputs[j]))
                            .sum());
                });
        return totalError;
    }


    // Calculates the mean of all the samples errors
    private static double calcMean(List<Double> totalError) {
        return totalError
                .stream()
                .reduce(0.0, Double::sum) / totalError.size();
    }

    // Calculates the standard deviation of the provided errors
    private static double calcStandardDeviation(double average, List<Double> totalError) {
        return Math.sqrt(totalError
                .stream()
                .mapToDouble(aDouble -> Math.pow((aDouble - average), 2) / totalError.size())
                .sum());
    }

    // Writes the mean ans standard deviation to std out
    private static void printStats(double mean, double standardDeviation, NetworkTrainerType networkType) {
        System.out.println(" ");
        System.out.println("Type: " + networkType.toString());
        System.out.println("-------------------------------");
        System.out.println("Mean error:         " + mean);
        System.out.println("Standard Deviation: " + standardDeviation);
    }

    private static void writeFile(String outputPath, List<String> lines) {
        try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(outputPath))) {
            writer.write(lines.stream()
                    .reduce((sum, currLine) -> sum + "\n" + currLine)
                    .orElse(""));
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    private static void logStep(int k, NetworkTrainerType trainerType) {
        setConsoleOut();
        System.out.println("k: " + k + ",\t" + trainerType.toString() + " complete.");
        setFileOut();
    }

    private static void setFileOut() {
        try {
            System.setOut(new PrintStream(new BufferedOutputStream(new FileOutputStream(datasetType.name().concat(".txt"), true)), true));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    private static void setConsoleOut() {
        System.setOut(new PrintStream(new FileOutputStream(FileDescriptor.out)));
    }
}


