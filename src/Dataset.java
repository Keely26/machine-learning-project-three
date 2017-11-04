import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Dataset extends ArrayList<Sample> {

    public Dataset() {

    }

    public Dataset(List<Sample> samples) {

    }


    public void shuffle() {
        Collections.shuffle(this);
    }

    public Dataset trainSet() {
        return new Dataset(this.subList(0, this.size() / 2));
    }
}
