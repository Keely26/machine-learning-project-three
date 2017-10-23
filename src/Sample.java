import java.util.Random;

/**
 * Sample class acts as a container to hold a set of inputs and their corresponding outputs
 * for the function being analyzed
 */
public class Sample {
    public double[] inputs;
    public double[] outputs;

    public Sample(int numInputs) {
        this.inputs = new double[numInputs];
        this.outputs = new double[1];
    }

    public Sample(double[] inputs, double[] outputs) {
        this.inputs = inputs;
        this.outputs = outputs;
    }

    public Sample(int numInputs, int maxInputVal, int numOutputs) {
        this.inputs = new double[numInputs];
        this.outputs = new double[numOutputs];

        Random random = new Random();
        for (int i = 0; i < numInputs; i++) {
            inputs[i] = random.nextDouble() * maxInputVal;
        }
    }
}
