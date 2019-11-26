import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Random;


public class J48 {

    public void run (String options, Datasets datasets)
    {
        try
        {
            String path_train = "", path_test = "";
            DataSource source_train = null, source_test = null;
            Instances data_train = null, data_test = null;

            switch (datasets) {
                case KFOLD:
                    path_train = getClass().getResource("trainingData_top10.csv").getPath();
                    source_train = new DataSource(path_train);
                    data_train = source_train.getDataSet();
                    data_train.setClassIndex(data_train.numAttributes() - 1);
                    break;
                case TEST_FULL:
                    path_train = getClass().getResource("trainingData_top10.csv").getPath();
                    source_train = new DataSource(path_train);
                    data_train = source_train.getDataSet();
                    data_train.setClassIndex(data_train.numAttributes() - 1);

                    path_test = getClass().getResource("testingData_top10.csv").getPath();
                    source_test = new DataSource(path_test);
                    data_test = source_test.getDataSet();
                    data_test.setClassIndex(data_test.numAttributes() - 1);
                    break;
                case TEST_4000:
                    path_train = getClass().getResource("trainingData_top10_4000.csv").getPath();
                    source_train = new DataSource(path_train);
                    data_train = source_train.getDataSet();
                    data_train.setClassIndex(data_train.numAttributes() - 1);

                    path_test = getClass().getResource("testingData_top10_4000.csv").getPath();
                    source_test = new DataSource(path_test);
                    data_test = source_test.getDataSet();
                    data_test.setClassIndex(data_test.numAttributes() - 1);
                    break;
                case TEST_9000:
                    path_train = getClass().getResource("trainingData_top10_9000.csv").getPath();
                    source_train = new DataSource(path_train);
                    data_train = source_train.getDataSet();
                    data_train.setClassIndex(data_train.numAttributes() - 1);

                    path_test = getClass().getResource("testingData_top10_9000.csv").getPath();
                    source_test = new DataSource(path_test);
                    data_test = source_test.getDataSet();
                    data_test.setClassIndex(data_test.numAttributes() - 1);
                    break;
                default:
                    throw new Exception("You need to select a compatible data set name (KFOLD, TEST_FULL, TEST_4000, TEST_9000)");
            }

            System.out.println("Building classifier: " + datasets.toString() + " " + options);

            // Create new instance of scheme
            weka.classifiers.trees.J48 model = new weka.classifiers.trees.J48();
            model.setOptions(Utils.splitOptions(options));
            model.buildClassifier(data_train);

//            System.out.println("=== Classifier model (full training set) ===\n");
//            System.out.println(model);

            Evaluation eval = null;

            switch (datasets) {
                case KFOLD:
                    // Eval it with test data based on the test data
                    eval = new Evaluation(data_train);
                    eval.evaluateModel(model, data_train);
                    eval.crossValidateModel(model, data_train, 10, new Random(1));
                    break;
                case TEST_FULL:
                case TEST_9000:
                case TEST_4000:
                    eval = new Evaluation(data_train);
                    eval.evaluateModel(model, data_test);
                    break;
                default:
                    throw new Exception("You need to select a compatible data set name (KFOLD, TEST_FULL, TEST_4000, TEST_9000)");
            }

//            System.out.println(eval.toSummaryString("=== Summary ===\n", false));
//            System.out.println(eval.toClassDetailsString());
//            System.out.println(eval.toMatrixString("=== Confusion Matrix ===\n"));

            String output =
                    eval.toSummaryString("=== Summary ===\n", false) + "\n" +
                    eval.toClassDetailsString() + "\n" +
                    eval.toMatrixString("=== Confusion Matrix ===\n") + "\n" +
                    "=== Classifier model (full training set) ===\n" + "\n" +
                    model;

            BufferedWriter writer = new BufferedWriter(new FileWriter(getClass().getResource("/").getPath() + datasets.toString() + "_" + options + ".txt"));
            writer.write(output);
            writer.close();
        }
        catch (Exception e)
        {
            System.err.println(e);
            System.err.println(e.getMessage());
        }
    }

}
