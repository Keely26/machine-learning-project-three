/**
 * Sample class acts as a container to hold a set of inputs and their corresponding outputs
 * for the function being analyzed
 */
public class Sample {
    public final double[] inputs;
    public final double[] outputs;

    public Sample(double[] inputs, double[] outputs) {
        this.inputs = inputs;
        this.outputs = outputs;
    }

    public Sample(String[] inputs, String[] outputs) {
        this.inputs = new double[inputs.length];
        this.outputs = new double[outputs.length];

        for (int i = 0; i < inputs.length; i++) {
            this.inputs[i] = Double.parseDouble(inputs[i]);
        }

        for (int i = 0; i < outputs.length; i++) {
            this.outputs[i] = Double.parseDouble(outputs[i]);
        }
    }
}
