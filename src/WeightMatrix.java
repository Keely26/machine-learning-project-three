import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

@SuppressWarnings("WeakerAccess")
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
            List<Neuron> currentLayer = network.getLayer(i).getNeurons();
            for (int j = 0; j < currentLayer.size(); j++) {
                List<Double> currentWeights = this.getModifiedWeights(i, j);
                currentLayer.get(i).setWeights(currentWeights);
            }
        }

        return this.network;
    }

    private List<Double> getModifiedWeights(int layer, int neuron) {
        int startIndex = 0;
        int endIndex = 9999;

        // TODO: Figure out how to compute indices
        // Should be sum of pairwise products of layer dimensions preceding layer, maybe?
        return this.weights.subList(startIndex, endIndex);
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
