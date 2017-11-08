import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * The WeightMatrix class provides a serialized representation of an INeuralNetwork, at initialization the dimensions
 * and weights of the network provided are extracted and recorded, where the weights are a 1D list, trainers may then use
 * this representation to perform various opperation such as mutation and crossover between different WeightMatrices
 */
public class WeightMatrix implements Comparable {

    private static final double sigmaIncreaseFactor = 1.1;
    private static final double sigmaDecreaseFactor = 0.9;

    private List<Double> weights;
    private List<Double> sigmas;
    private List<Integer> dimensions;

    private double fitness;
    private INeuralNetwork network;

    private final int networkSize;
    private final int numInputs;

    public WeightMatrix(INeuralNetwork network) {
        this.network = network;
        this.networkSize = network.getSize();
        this.numInputs = network.getLayer(0).getNeuron(0).size;
        this.weights = new ArrayList<>();
        this.sigmas = new ArrayList<>();
        this.dimensions = new ArrayList<>();

        for (int i = 0; i < networkSize; i++) {
            Layer layer = network.getLayer(i);
            this.dimensions.add(layer.size);
            this.weights.addAll(layer
                    .getNeurons()
                    .stream()
                    .map(Neuron::getWeights)
                    .flatMap(Collection::stream)
                    .collect(Collectors.toList()));
        }
    }

    public WeightMatrix(INeuralNetwork network, List<Double> weights) {
        this.network = network;
        this.networkSize = network.getSize();
        this.numInputs = network.getLayer(0).getNeuron(0).size;
        this.weights = new ArrayList<>(weights);
        this.dimensions = new ArrayList<>();

        for (int i = 0; i < networkSize; i++) {
            this.dimensions.add(network.getLayer(i).size);
        }
    }

    /**
     * Construct an INeuralNetwork from the dimensions and weights stored in the class
     */
    public INeuralNetwork buildNetwork() {
        for (int i = 0; i < networkSize; i++) {
            List<Neuron> currentLayer = network.getLayer(i).getNeurons();
            for (int j = 0; j < currentLayer.size(); j++) {
                List<Double> currentWeights = this.getModifiedWeights(i, j);
                currentLayer.get(j).setWeights(currentWeights);
            }
        }

        return this.network;
    }

    /**
     * Retrieve the list of weights corresponding to a particular neuron in a particular layer
     */
    private List<Double> getModifiedWeights(int layer, int neuron) {
        int startIndex, endIndex;

        // Accumulate layer index offset
        startIndex = IntStream
                .range(0, layer)
                .parallel()
                .map(i -> i == 0 ? numInputs * dimensions.get(0) : dimensions.get(i - 1) * dimensions.get(i))
                .sum();

        // Accumulate neuron index offset, set end index
        if (layer == 0) {
            startIndex += neuron * numInputs;
            endIndex = startIndex + numInputs;
        } else {
            startIndex += neuron * this.dimensions.get(layer - 1);
            endIndex = startIndex + this.dimensions.get(layer - 1);
        }

        return new ArrayList<>(this.weights.subList(startIndex, endIndex));
    }

    public List<Double> getWeights() {
        return this.weights;
    }

    public void setWeights(List<Double> weights) {
        assert weights.size() == this.weights.size() : "Invalid weight matrix";

        this.weights = weights;
    }

    public double getFitness() {
        return this.fitness;
    }

    public void setFitness(double fitness) {
        this.fitness = fitness;
    }

    /**
     * Modify each sigma according the the 1/5th rule, most fit individuals have their sigma decreased, while other
     * are increased according to the constant factors set above
     */
    public void adjustSigma(boolean increase) {
        IntStream.range(0, this.sigmas.size())
                .parallel()
                .forEach(i -> this.sigmas.set(i, this.sigmas.get(i) * (increase ? sigmaIncreaseFactor : sigmaDecreaseFactor)));
    }

    public List<Double> getSigmas() {
        return sigmas;
    }

    public void setSigmas(List<Double> sigmas) {
        this.sigmas = sigmas;
    }

    @Override
    public int compareTo(Object o) {
        return Double.compare(fitness, ((WeightMatrix) o).fitness);
    }
}
