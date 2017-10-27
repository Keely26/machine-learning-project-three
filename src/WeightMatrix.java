import java.util.List;

public class WeightMatrix {

    private List<Double> weights;
    private List<Integer> dimmensions;

    private INeuralNetwork network;


    public WeightMatrix(INeuralNetwork network) {
        this.network = network;
        int numLayers = network.getSize();

        for (int i = 0; i < numLayers; i++) {
            Layer layer = network.getLayer(i);
        }
    }


    public INeuralNetwork buildNetwork() {
        // Construct and set new layers with current weights
        return this.network;
    }
}
