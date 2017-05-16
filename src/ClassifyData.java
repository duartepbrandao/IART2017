import weka.classifiers.Evaluation;

import weka.classifiers.trees.I
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.core.*;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class ClassifyData {

    static Id3 classifier;
    static Evaluation evaluation;
   public static Instances fileInstances, train, evaluate;

    public ClassifyData() throws Exception{
        classifier = new J48();

    }

    public void setInstances(String filePath, int percentage){

        DataSource source = null;
        try { source = new DataSource(filePath);
        } catch (Exception e) {
            e.printStackTrace();
        }
        try {
            fileInstances = source.getDataSet();
        } catch (Exception e) {
            e.printStackTrace();
        }
        int trainSize = Math.round(fileInstances.numInstances()*percentage/100);
        train = new Instances(fileInstances,0,trainSize);
        int evaluateSize = fileInstances.numInstances() - trainSize;
        evaluate = new Instances(fileInstances,trainSize,evaluateSize);
    }

    public void train() throws Exception {

        train.setClassIndex(train.numAttributes() - 1);

        NumericToNominal convert= new NumericToNominal();
        String[] options= new String[2];
        options[0]="-R";
        options[1]="first-last";  //range of variables to make numeric

        convert.setOptions(options);
        convert.setInputFormat(train);

        Instances newData=Filter.useFilter(train, convert);
        train =newData;

        classifier.buildClassifier(train);
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(classifier,train);

        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
    }


}
