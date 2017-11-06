import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Neuron class stores values weights and activations of a single neuron in the network and provides functionality
 * to compute the output of the that neuron.
 */
@SuppressWarnings("WeakerAccess")
public class Neuron {

    public final int size;

    private List<Double> weights;
    private List<Double> previousWeights;
    private double activation;
    private double delta;
    private double bias;

    private IActivationFunction activationFunction;

    // Feed forward network constructor
    public Neuron(int connections, IActivationFunction activationFunction) {
        this.size = connections;
        this.activationFunction = activationFunction;
        this.initializeWeights();
    }

    // Set up connection weights, set to random value between [-0.5, 0.5] or all 1 depending on flag
    private void initializeWeights() {
        Random random = new Random(System.nanoTime());
        this.weights = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            this.weights.add(random.nextDouble() - 0.00005);
        }
        this.bias = random.nextDouble() - 0.00005;
        this.previousWeights = new ArrayList<>(this.weights);
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

    public List<Double> getWeights() {
        return this.weights;
    }

    public void setWeights(List<Double> weights) {
        this.weights = weights;
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

    // Set the previous weight as the current weight and increment the current weight by the supplied value
    public void updateWeight(int index, double increment) {
        double previousWeight = this.weights.get(index);
        this.previousWeights.set(index, previousWeight);
        this.weights.set(index, previousWeight + increment);
    }

    public double getPreviousWeight(int index) {
        return this.previousWeights.get(index);
    }
}