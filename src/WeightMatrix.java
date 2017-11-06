import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class WeightMatrix implements Comparable {

    private List<Double> weights;
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

    private List<Double> getModifiedWeights(int layer, int neuron) {
        int startIndex, endIndex;

        // Accumulate layer index offset
        startIndex = IntStream
                .range(0, layer)
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

    @Override
    public int compareTo(Object o) {
        return Double.compare(fitness, ((WeightMatrix) o).fitness);
    }
}
