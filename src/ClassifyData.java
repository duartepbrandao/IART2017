import weka.classifiers.Evaluation;

import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.core.*;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;

import java.util.Random;

public class ClassifyData {

    static J48 classifier;
    static Evaluation evaluation;
    public static Instances fileInstances, train, evaluate;

    public ClassifyData() throws Exception {
        classifier = new J48();
        classifier.setBinarySplits(false);
        classifier.setCollapseTree(true);
        classifier.setUnpruned(false);
        classifier.setConfidenceFactor((float) 0.25);
        classifier.setMinNumObj(3);
        classifier.setNumFolds(3);
    }

    public void setInstances(String filePath, int percentage) {

        DataSource source = null;
        try {
            source = new DataSource(filePath);
        } catch (Exception e) {
            e.printStackTrace();
        }
        try {
            fileInstances = source.getDataSet();
        } catch (Exception e) {
            e.printStackTrace();
        }
        int trainSize = Math.round(fileInstances.numInstances() * percentage / 100);


        train = new Instances(fileInstances, 0, trainSize);
        int evaluateSize = fileInstances.numInstances() - trainSize;
        System.out.println("trainSize = " + trainSize);
        System.out.println("fileInstances = " + fileInstances.numInstances());
        System.out.println("evaluate = " + evaluateSize);
        evaluate = new Instances(fileInstances, trainSize, evaluateSize);
    }

    public void train() throws Exception {



        NumericToNominal convert = new NumericToNominal();
        String[] options = new String[2];
        options[0] = "-R";
        options[1] = "last";  //range of variables to make numeric

        convert.setOptions(options);
        convert.setInputFormat(train);

        Instances newData = Filter.useFilter(train, convert);
        train = newData;

        train.setClassIndex(train.numAttributes() - 1);
        classifier.buildClassifier(train);
    }

    public void test() throws Exception {

        NumericToNominal convert = new NumericToNominal();
        String[] options = new String[2];
        options[0] = "-R";
        options[1] = "last";  //range of variables to make numeric
        Instances newData = null;

        convert.setOptions(options);
        convert.setInputFormat(evaluate);

        newData = Filter.useFilter(evaluate, convert);

        evaluate = newData;

        evaluate.setClassIndex(evaluate.numAttributes() - 1);
        evaluation = new Evaluation(train);

        //need to change train to evaluate
        evaluation.crossValidateModel(classifier, evaluate,10, new Random());

        System.out.println(evaluation.toSummaryString("\nResults\n======\n", false));
        evaluation.evaluateModel(classifier,evaluate);

        System.out.println(evaluation.toSummaryString("\nResults\n======\n", false));

    }


    public void xp() throws Exception {
        Remove rm = new Remove();
        rm.setAttributeIndices("1");  // remove 1st attribute
        // classifier
        J48 j48 = new J48();
        j48.setUnpruned(true);        // using an unpruned J48
        // meta-classifier
        evaluate.setClassIndex(evaluate.numAttributes() - 1);
        FilteredClassifier fc = new FilteredClassifier();
        fc.setFilter(rm);
        fc.setClassifier(j48);
        // train and make predictions
        fc.buildClassifier(train);
        for (int i = 0; i < evaluate.numInstances(); i++) {
            double pred = fc.classifyInstance(evaluate.instance(i));
            System.out.print("ID: " + evaluate.instance(i).value(0));
            System.out.print(", actual: " + evaluate.classAttribute().value((int) evaluate.instance(i).classValue()));
            System.out.println(", predicted: " + evaluate.classAttribute().value((int) pred));


        }

    }
}
