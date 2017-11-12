import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/**
 * An implementation of standard backpropagation
 * Supports stochastic weight updates and momentum
 *
 * @author Zach Connelly
 */
public class BPNetworkTrainer extends NetworkTrainerBase {

    private double learningRate;
    private double momentum;
    private int batchSize;

    private int bestIteration;
    private double bestError = Double.MAX_VALUE;
    private WeightMatrix bestNetwork;

    BPNetworkTrainer(double learningRate, double momentum, int batchSize) {
        super(0);
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.batchSize = batchSize;
    }

    @Override
    public INeuralNetwork train(INeuralNetwork network, Dataset samples) {
        startTimer();
        // Split training set into training and validation sets
        samples.shuffle();
        Dataset validationSet = new Dataset(samples.subList(0, samples.size() / 10));
        Dataset trainingSet = new Dataset(samples.subList(samples.size() / 10, samples.size()));

        int iteration = 0;
        while (shouldContinue(validate(network, validationSet, iteration), iteration, network)) {
            for (Sample sample : trainingSet) {
                // Forward propagate each sample through the network
                network.execute(sample.inputs);

                // Backpropagate the error using the true outputs
                this.backPropagate(network, sample.outputs);

                // Only apply weight updates once per batch
                if (iteration % batchSize == 0) {
                    this.updateWeights(network, sample.inputs);
                    this.resetWeightDeltas(network);
                }
            }
            iteration++;
        }

        printConvergence(NetworkTrainerType.BPNetworkTrainer, network);
        return bestNetwork.buildNetwork();
    }

    /**
     * Determine if learning should be cutoff by evaluating performance against the validation set
     */
    protected boolean shouldContinue(double validationError, int iteration, INeuralNetwork network) {
        if (validationError < bestError) {
            bestIteration = iteration;
            bestError = validationError;
            bestNetwork = network.constructWeightMatrix();
        }
        runningAvg = ((runningAvg * 9) + validationError) / 10;
        // Increment counter if the error has seen less than 1/10000 improvement
        if ((runningAvg - validationError) / validationError < 0.0001) {
            cutoffCounter++;
        } else {
            cutoffCounter = 0;
        }
        return iteration < 10000 && cutoffCounter < 500;
    }

    /**
     * Compute the weight deltas for each weight in the network starting with the output layer
     * Use the gradient of the output errors to compute the deltas for the first hidden layer
     * Use the deltas established in the first hidden layer to compute those for subsequent layers
     */
    private void backPropagate(INeuralNetwork network, double[] expectedOutputs) {
        for (int i = network.getSize() - 1; i >= 0; i--) {
            Layer currentLayer = network.getLayer(i);
            List<Double> errors = new ArrayList<>();

            if (i != network.getSize() - 1) {
                // Compute errors for hidden layers
                for (int j = 0; j < currentLayer.size; j++) {
                    double error = 0.0;
                    Layer prevLayer = network.getLayer(i + 1);
                    for (int k = 0; k < prevLayer.size; k++) {
                        error += prevLayer.getNeuron(k).getWeight(j) * prevLayer.getNeuron(k).getDelta();
                    }
                    errors.add(error);
                }
            } else {
                // Compute errors for output layer
                for (int j = 0; j < currentLayer.size; j++) {
                    Neuron neuron = currentLayer.getNeuron(j);
                    double error = expectedOutputs[j] - neuron.getOutput();
                    errors.add(error);
                }
            }

            // Set deltas by multiplying the partial error by the derivative of the respective output
            for (int j = 0; j < currentLayer.size; j++) {
                Neuron neuron = currentLayer.getNeuron(j);
                double delta = errors.get(j) * network.computeActivationDerivative(neuron.getOutput());
                neuron.setDelta(delta);
            }
        }
    }

    /**
     * Use the weight deltas calculated by back prop to update the weights of each neuron
     */
    private void updateWeights(INeuralNetwork network, double[] networkInputs) {
        double[] inputs;
        for (int i = 0; i < network.getSize(); i++) {
            // Set inputs to the layer in question as the outputs of the previous layer
            if (i == 0) {
                inputs = networkInputs;
            } else {
                Layer prevLayer = network.getLayer(i - 1);
                inputs = new double[prevLayer.size];
                for (int j = 0; j < prevLayer.size; j++) {
                    inputs[j] = prevLayer.getNeuron(j).getOutput();
                }
            }
            // Apply weight updates to each neuron in the current layer and update the bias node
            Layer currentLayer = network.getLayer(i);
            for (int j = 0; j < currentLayer.size; j++) {
                Neuron currentNeuron = currentLayer.getNeuron(j);
                for (int k = 0; k < inputs.length; k++) {
                    // Apply momentum
                    double updatedWeight = ((1 - this.momentum) * this.learningRate * currentNeuron.getDelta() * inputs[k]) +
                            (momentum * currentNeuron.getPreviousWeight(k));
                    currentNeuron.updateWeight(k, updatedWeight);
                }
                currentNeuron.updateBias(this.learningRate * currentNeuron.getDelta());
            }
        }
    }

    /**
     * Reset all of the weight deltas in the network to zero
     */
    private void resetWeightDeltas(INeuralNetwork network) {
        IntStream.range(0, network.getSize())
                .forEach(i -> network.getLayer(i)
                        .getNeurons()
                        .forEach(neuron -> neuron.setDelta(0.0)));
    }

    /**
     * Compute the average error over the validation set and print to console
     */
    private double validate(INeuralNetwork network, Dataset validationSet, int epoch) {
        double error = validationSet
                .parallelStream()
                .mapToDouble(sample -> this.meanSquaredError(network.execute(sample.inputs), sample.outputs))
                .sum() / validationSet.size();
        if (epoch % 10 == 0) {
            System.out.println("Epoch: " + epoch + "\t\tValidation Set Error: " + error);
        }
        return error;
    }
}
