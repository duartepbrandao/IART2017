import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.converters.ConverterUtils.DataSource;

import weka.core.*;

public class ClassifyData {

    static MultilayerPerceptron classifier;
    static Evaluation evaluation;
    static Instances fileInstances, train, test;

    public ClassifyData() throws Exception{
        classifier = new MultilayerPerceptron();
        classifier.setOptions(Utils.splitOptions("-L 0.4 -M 0.2 -N 700 -V 0 -S 0 -E 20 -H 17"));
    }

    public getInstances(String filePath, int percentage){

        DataSource source = null;
        try { source = new DataSource(filePath);
        } catch (Exception e) {
            e.printStackTrace();
        }
        fileInstances = source.getDataSet()
    }

}
