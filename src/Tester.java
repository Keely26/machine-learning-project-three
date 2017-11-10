
import java.io.BufferedWriter;
import java.io.IOException;
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


    public static void main(String[] args) {
        INeuralNetwork MLP = NetworkFactory.buildNewNetwork(NetworkType.MultiLayerPerceptron);

        INetworkTrainer GA = NetworkFactory.buildNetworkTrainer(NetworkTrainerType.GANetworkTrainer);
        INetworkTrainer ES = NetworkFactory.buildNetworkTrainer(NetworkTrainerType.ESNetworkTrainer);
        INetworkTrainer DE = NetworkFactory.buildNetworkTrainer(NetworkTrainerType.DENetworkTrainer);
        INetworkTrainer BP = NetworkFactory.buildNetworkTrainer(NetworkTrainerType.BPNetworkTrainer);

        List<INetworkTrainer> trainers = new ArrayList<>();
        trainers.add(GA);
        trainers.add(ES);
        trainers.add(DE);
        trainers.add(BP);

        //Dataset trainingSet = DatasetFactory.buildDataSet("ecoli");
        //Dataset trainingSet = DatasetFactory.buildDataSet("energy");
        Dataset trainingSet = DatasetFactory.buildDataSet("tic-tac-toe");
        //Dataset trainingSet = DatasetFactory.buildDataSet("winequality");
        //Dataset trainingSet = DatasetFactory.buildDataSet("yeast");


        assert trainingSet != null;

        //trainer.train(MLP, trainingSet);

        crossValidate(trainers, trainingSet, MLP);
    }

    // Execute a 5x2 cross validation for both networks computing the mean and standard deviation of their errors
    public static void crossValidate(List<INetworkTrainer> trainers, Dataset dataset, INeuralNetwork network) {

        Dataset dataSet = DatasetFactory.buildDataSet("tic-tac-toe");

        List<Double> ESErrors = new ArrayList<>();
        List<Double> DEErrors = new ArrayList<>();
        List<Double> GAErrors = new ArrayList<>();
        List<Double> BPErrors = new ArrayList<>();

        for (int k = 0; k < 5; k++) {
            //System.setOut();
            Collections.shuffle(dataSet);
            Dataset testSet = dataset.getTestingSet();
            Dataset trainSet = dataset.getTrainingSet();

            System.out.println("trainer: " + trainers.get(0));
            GAErrors.addAll(computeFold(trainSet, testSet, network, trainers.get(0)));

            network = NetworkFactory.buildNewNetwork(NetworkType.MultiLayerPerceptron);
            System.out.println("trainer: " + trainers.get(1));
            ESErrors.addAll(computeFold(trainSet, testSet, network, trainers.get(1)));

            network = NetworkFactory.buildNewNetwork(NetworkType.MultiLayerPerceptron);
            System.out.println("trainer: " + trainers.get(2));
            DEErrors.addAll(computeFold(trainSet, testSet, network, trainers.get(2)));

            network = NetworkFactory.buildNewNetwork(NetworkType.MultiLayerPerceptron);
            System.out.println("trainer: " + trainers.get(3));
            BPErrors.addAll(computeFold(trainSet, testSet, network, trainers.get(3)));


            //training set and test set are swapped
            network = NetworkFactory.buildNewNetwork(NetworkType.MultiLayerPerceptron);
            System.out.println("S trainer: " + trainers.get(0));
            GAErrors.addAll(computeFold(testSet, trainSet, network, trainers.get(0)));

            network = NetworkFactory.buildNewNetwork(NetworkType.MultiLayerPerceptron);
            System.out.println("S trainer: " + trainers.get(1));
            ESErrors.addAll(computeFold(testSet, trainSet, network, trainers.get(1)));

            network = NetworkFactory.buildNewNetwork(NetworkType.MultiLayerPerceptron);
            System.out.println("S trainer: " + trainers.get(2));
            DEErrors.addAll(computeFold(testSet, trainSet, network, trainers.get(2)));

            network = NetworkFactory.buildNewNetwork(NetworkType.MultiLayerPerceptron);
            System.out.println("S trainer: " + trainers.get(3));
            BPErrors.addAll(computeFold(testSet, trainSet, network, trainers.get(3)));
        }

        double mean = calcMean(GAErrors);
        double SD = calcStandardDeviation(mean, GAErrors);
        printStats(mean, SD, "GA");

        mean = calcMean(ESErrors);
        SD = calcStandardDeviation(mean, ESErrors);
        printStats(mean, SD, "ES");

        mean = calcMean(DEErrors);
        SD = calcStandardDeviation(mean, DEErrors);
        printStats(mean, SD, "DE");

        mean = calcMean(BPErrors);
        SD = calcStandardDeviation(mean, BPErrors);
        printStats(mean, SD, "BP");

    }

    private static List<Double> computeFold(Dataset trainSet, Dataset testSet, INeuralNetwork network, INetworkTrainer trainer) {
        trainer.train(network, trainSet);
        return getApproximationErrors(testSet, network);
    }

    // Iterates through testing set and calculates the approximated values and the error of the samples of the supplied network
    private static List<Double> getApproximationErrors(List<Sample> testSet, INeuralNetwork network) {
        List<Double> totalError = new ArrayList<>(testSet.size());
        for (Sample sample : testSet) {
            // Get the network's approximation
            double[] networkOutput = network.execute(sample.inputs);

            // Add the error for each sample to the total error
            totalError.add(IntStream
                    .range(0, networkOutput.length)
                    .mapToDouble(j -> Math.abs(networkOutput[j] - sample.outputs[j]))
                    .sum());
        }
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
    private static void printStats(double mean, double standardDeviation, String networkType) {
        System.out.println(" ");
        System.out.println("Network type: " + networkType);
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
}


