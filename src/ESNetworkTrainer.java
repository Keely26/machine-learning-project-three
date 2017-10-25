import java.util.List;

public class ESNetworkTrainer extends NetworkTrainerBase {

    ESNetworkTrainer() {

    }

    @Override
    public INeuralNetwork train(INeuralNetwork network, List<Sample> samples) {
        return network;
    }
}
