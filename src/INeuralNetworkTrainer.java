import java.util.List;

public interface INeuralNetworkTrainer {
    INeuralNetwork train(INeuralNetwork network, List<Sample> samples);
}
