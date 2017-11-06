import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class BPNetworkTrainer extends NetworkTrainerBase {

    private double learningRate;
    private double momentum;
    private int batchSize;
    private int epochs;

    BPNetworkTrainer(double learningRate, double momentum, int batchSize, int epochs) {
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.batchSize = batchSize;
        this.epochs = epochs;
    }

    @Override
    public INeuralNetwork train(INeuralNetwork network, Dataset samples) {
        // Iterate over the defined number of epochs
        for (int i = 0; i < epochs; i++) {
            double epochError = 0.0;
            for (Sample sample : samples) {
                // Forward propagate each sample through the network
                double[] networkOutputs = this.execute(network, sample.inputs);

                // Backpropagate the error using the true outputs
                this.backPropagate(network, sample.outputs);

                // Only apply weight updates once per batch
                if (i % batchSize == 0) {
                    this.updateWeights(network, sample.inputs);
                    this.resetWeightDeltas(network);
                }
                // Sum the total error for each iteration
                epochError += this.meanSquaredError(sample.outputs, networkOutputs);
            }
            System.out.println("Epoch: " + i + "\t\tError: " + epochError / samples.size());
        }

        return network;
    }

    // Compute the weight deltas for each weight in the network starting with the output layer
    // Use the gradient of the output errors to compute the deltas for the first hidden layer
    // Use the deltas established in the first hidden layer to compute those for subsequent layers
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

    // Use the weight deltas calculated by back prop to update the weights of each neuron
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

    // TODO: Extract weight deltas from each neuron into matrix in trainer class
    // Reset all of the weight deltas in the network to zero
    private void resetWeightDeltas(INeuralNetwork network) {
        IntStream.range(0, network.getSize())
                .forEach(i -> network.getLayer(i)
                        .getNeurons()
                        .forEach(neuron -> neuron.setDelta(0.0)));
    }
}
