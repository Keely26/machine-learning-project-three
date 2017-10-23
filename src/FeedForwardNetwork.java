import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;

public class FeedForwardNetwork extends NeuralNetwork {

    private List<Layer> network;

    private double learningRate;
    private double momentum;

    private int batchSize;
    private int epochs;

    public FeedForwardNetwork(int[] networkDimensions, double learningRate, int batchSize, double momentum, IActivationFunction activationFunction, int epochs) {
        super(networkDimensions[0], networkDimensions[networkDimensions.length - 1]);
        this.learningRate = learningRate;
        this.batchSize = batchSize;
        this.momentum = momentum;
        this.epochs = epochs;
        this.activationFunction = activationFunction;

        this.initializeNetwork(networkDimensions);
    }

    @Override
    public void train(List<Sample> samples) {

        // Iterate over the defined number of epochs
        for (int i = 0; i < epochs; i++) {
            double epochError = 0.0;
            for (Sample sample : samples) {
                // Forward propagate each sample through the network
                double[] networkOutputs = this.forwardPropagate(sample.inputs);

                // Backpropagate the error using the true outputs
                this.backPropagate(sample.outputs);

                // Only apply weight updates once per batch
                if (i % batchSize == 0) {
                    this.updateWeights(sample.inputs);
                    this.resetWeightDeltas();
                }
                // Sum the total error for each iteration
                epochError += this.calculateTotalError(sample.outputs, networkOutputs);
            }
            System.out.println("Epoch: " + i + "\t\tError: " + epochError / samples.size());
        }
    }

    @Override
    public double[] approximate(double[] inputs) {
        // Forward propagate inputs through the network returning the output of the final layer
        return this.forwardPropagate(inputs);
    }

    private double[] forwardPropagate(double[] inputs) {
        double[] layerOutputs = inputs;

        // Propagate through hidden layers, not using the activation function for the final layer
        for (int i = 0; i < network.size(); i++) {
            boolean shouldActivate = (i != network.size() - 1);
            layerOutputs = network.get(i).execute(layerOutputs, shouldActivate);
        }

        return layerOutputs;
    }

    // Compute the weight deltas for each weight in the network starting with the output layer
    // Use the gradient of the output errors to compute the deltas for the first hidden layer
    // Use the deltas established in the first hidden layer to compute those for subsequent layers
    private void backPropagate(double[] expectedOutputs) {
        for (int i = this.network.size() - 1; i >= 0; i--) {
            Layer currentLayer = this.network.get(i);
            List<Double> errors = new ArrayList<>();

            if (i != this.network.size() - 1) {
                // Compute errors for hidden layers
                for (int j = 0; j < currentLayer.size; j++) {
                    double error = 0.0;
                    Layer prevLayer = this.network.get(i + 1);
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
                double delta = errors.get(j) * this.activationFunction.computeDerivative(neuron.getOutput());
                neuron.setDelta(delta);
            }
        }
    }

    // Use the weight deltas calculated by back prop to update the weights of each neuron
    private void updateWeights(double[] networkInputs) {
        double[] inputs;
        for (int i = 0; i < this.network.size(); i++) {
            // Set inputs to the layer in question as the outputs of the previous layer
            if (i == 0) {
                inputs = networkInputs;
            } else {
                Layer prevLayer = this.network.get(i - 1);
                inputs = new double[prevLayer.size];
                for (int j = 0; j < prevLayer.size; j++) {
                    inputs[j] = prevLayer.getNeuron(j).getOutput();
                }
            }
            // Apply weight updates to each neuron in the current layer and update the bias node
            Layer currentLayer = this.network.get(i);
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

    // Reset all of the weight deltas in the network to zero
    private void resetWeightDeltas() {
        this.network.forEach(layer -> layer.getNeurons().forEach(neuron -> neuron.setDelta(0.0)));
    }

    // Initialize each layer of the network, in the input layer is created implicitly in execution and does not have a layer
    private void initializeNetwork(int[] networkDimensions) {
        this.network = new ArrayList<>(networkDimensions.length);
        for (int i = 1; i < networkDimensions.length; i++) {
            this.network.add(new Layer(networkDimensions[i], networkDimensions[i - 1], this.activationFunction));
        }
    }
}