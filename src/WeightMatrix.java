import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

public class WeightMatrix {

    private List<Double> weights;
    private List<Integer> dimmensions;

    private INeuralNetwork network;
    private final int networkSize;


    public WeightMatrix(INeuralNetwork network) {
        this.network = network;
        this.networkSize = network.getSize();
        this.weights = new ArrayList<>();

        for (int i = 0; i < networkSize; i++) {
            Layer layer = network.getLayer(i);
            this.dimmensions.add(layer.size);
            this.weights.addAll(layer.getNeurons()
                    .stream()
                    .map(Neuron::getWeights)
                    .flatMap(Collection::stream)
                    .collect(Collectors.toList())
            );
        }
    }


    public INeuralNetwork buildNetwork() {
        for (int i = 0; i < networkSize; i++) {

        }

        return this.network;
    }

    public List<Double> getWeights() {
        return this.weights;
    }

    public void setWeights(List<Double> weights) {
        if (weights.size() != this.weights.size()) {
            throw new IllegalArgumentException("Invalid weight matrix");
        }

        this.weights = weights;
    }
}
