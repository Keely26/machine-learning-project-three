import java.util.List;

public interface INeuralNetwork {
    void train(List<Sample> samples);

    double[] approximate(double[] inputs);

    double[][] serialize();

    void deserialize(double[][] weightMatrix);
}
