import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Random;


public class RandomForest {

    public void run (String options)
    {
        try
        {
            String path = getClass().getResource("data_top_10.csv").getPath();
            DataSource source = new DataSource(path);
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);

            // Create new instance of scheme
            weka.classifiers.trees.RandomForest model = new weka.classifiers.trees.RandomForest();
            model.setOptions(Utils.splitOptions(options));
            model.buildClassifier(data);

            System.out.println("=== Classifier model (full training set) ===\n");
            System.out.println(model);

            // Eval it with test data based on the test data
            Evaluation eval = new Evaluation(data);
            eval.evaluateModel(model, data);
            eval.crossValidateModel(model, data, 10, new Random(1));

            System.out.println(eval.toSummaryString("=== Summary ===\n", false));
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString("=== Confusion Matrix ===\n"));
        }
        catch (Exception e)
        {
            System.err.println(e);
            System.err.println(e.getMessage());
        }
    }

}
