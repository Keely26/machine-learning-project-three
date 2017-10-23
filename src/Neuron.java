import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Neuron class stores values weights and activations of a single neuron in the network and provides functionality
 * to compute the output of the that neuron. Supports use cases for both RBF and MLP
 */
public class Neuron {

    public final int size;

    private List<Double> weights;
    private List<Double> previousWeights;
    private double activation;
    private double delta;
    private double bias;

    private double[] inputs;
    private double[] outputs;
    private double[] mean;

    private IActivationFunction activationFunction;

    // Feed forward network constructor
    public Neuron(int connections, IActivationFunction activationFunction) {
        this.size = connections;
        this.activationFunction = activationFunction;
        this.initializeWeights(true);
    }

    // Radial basis network constructor
    public Neuron(int connections, boolean randomizeWeights) {
        this.size = connections;
        this.initializeWeights(randomizeWeights);
    }

    // Set up connection weights, set to random value between [-0.5, 0.5] or all 1 depending on flag
    private void initializeWeights(boolean randomizeWeights) {
        Random random = new Random(System.nanoTime());
        this.weights = new ArrayList<>(size);
        if (randomizeWeights) {
            for (int i = 0; i < size; i++) {
                this.weights.add(random.nextDouble() - 0.5);
            }
            this.bias = random.nextDouble() - 0.5;
        } else {
            for (int i = 0; i < size; i++) {
                this.weights.add(1.0);
            }
            this.bias = 1.0;
        }
        this.previousWeights = new ArrayList<>(size);
        this.previousWeights.addAll(this.weights);
    }

    // Calculate the activation of the neuron given a set of inputs, apply activation function if flag is set
    public double execute(double[] inputs, boolean shouldUseActivationFunction) {
        double outputSum = bias;
        for (int i = 0; i < size; i++) {
            outputSum += inputs[i] * weights.get(i);
        }
        this.activation = shouldUseActivationFunction ? this.activationFunction.compute(outputSum) : outputSum;
        return activation;
    }

    public double getWeight(int index) {
        return this.weights.get(index);
    }

    public double getOutput() {
        return this.activation;
    }

    public double getDelta() {
        return this.delta;
    }

    public void setDelta(double delta) {
        this.delta = delta;
    }

    public void updateBias(double increment) {
        this.bias += increment;
    }

    // Set the previous weight as the current weight and increment the current weight by the suppied value
    public void updateWeight(int index, double increment) {
        double previousWeight = this.weights.get(index);
        this.previousWeights.set(index, previousWeight);
        this.weights.set(index, previousWeight + increment);
    }

    public double getPreviousWeight(int index) {
        return this.previousWeights.get(index);
    }

    public double[] getMean() {
        return mean;
    }

    public void setMean(double[] mean) {
        this.mean = mean;
    }

    public double[] getInputs() {
        return inputs;
    }

    public double getInput(int index) {
        return this.inputs[index];
    }

    public void setInputs(double[] inputs) {
        this.inputs = inputs;
    }

    public double[] getOutputs() {
        return outputs;
    }

    public void setOutputs(double[] outputs) {
        this.outputs = outputs;
    }
}