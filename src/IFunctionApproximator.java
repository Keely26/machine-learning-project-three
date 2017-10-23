import java.util.List;

/**
 * Interface enforcing functionality of different neural networks
 */
public interface IFunctionApproximator {
    void train(List<Sample> samples);
    double[] approximate(double[] inputs);
}
