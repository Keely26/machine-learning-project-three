import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;

import java.util.ArrayList;

import java.util.List;
import java.util.Objects;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;


public class DatasetFactory {


    private static final String PATH = "datasets/";
    private static final String CSV = ".csv";
    private static final Pattern COMMA_DELIMITER = Pattern.compile(", ");

    public static Dataset buildDataSet(String fileName) {
        File file = new File(PATH.concat(fileName).concat(CSV));
        List<String> lines = Objects.requireNonNull(getFileStream(file)).collect(Collectors.toList());

        String[] header = COMMA_DELIMITER.split(lines.remove(0));
        int features = Integer.parseInt(header[0]);
        int classes = Integer.parseInt(header[1]);

        Dataset samples = new Dataset();
        lines.forEach(line -> {
            String[] parts = COMMA_DELIMITER.split(line);
            String[] inputs = new String[features];
            String[] outputs = new String[classes];

            System.arraycopy(parts, 0, inputs, 0, features);
            System.arraycopy(parts, features, outputs, 0, classes);

            samples.add(new Sample(inputs, outputs));
        });

        return samples;
    }

    private static Stream<String> getFileStream(File file) {
        try {
            return new BufferedReader(new FileReader(file)).lines();
        } catch (FileNotFoundException ex) {
            ex.printStackTrace();
            return null;
        }
    }
}

