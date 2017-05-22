
import weka.classifiers.Evaluation;

import weka.classifiers.trees.J48;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.core.*;
import weka.filters.unsupervised.attribute.NumericToNominal;

import weka.gui.treevisualizer.PlaceNode1;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

import java.awt.*;
import java.io.File;
import java.io.PrintWriter;
import java.util.Random;

public class ClassifyData {

    static J48 classifier;
    static Evaluation evaluation;
    public static Instances fileInstances, train, evaluate;

    public ClassifyData() throws Exception {
        classifier = new J48();
        classifier.setConfidenceFactor((float) 0.25);
        classifier.setUnpruned(true);
        classifier.setMinNumObj(2);
        classifier.setNumFolds(2);
        classifier.setCollapseTree(false);
        classifier.setSubtreeRaising(true);
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
            fileInstances.randomize(new Random());
        } catch (Exception e) {
            e.printStackTrace();
        }
        int trainSize = Math.round(fileInstances.numInstances() * percentage / 100);


        train = new Instances(fileInstances, 0, trainSize);
        int evaluateSize = fileInstances.numInstances() - trainSize;
        System.out.println("fileInstances = " + fileInstances.numInstances());
        System.out.println("trainSize = " + trainSize);
        System.out.println("evaluate = " + evaluateSize);
        evaluate = new Instances(fileInstances, trainSize, evaluateSize);
    }

    public void train() throws Exception {


        NumericToNominal convert = new NumericToNominal();
        String[] options = new String[2];
        options[0] = "-R";
        options[1] = "last";  //range of variables to make numeric


        train.setClassIndex(train.numAttributes() - 1);
        convert.setOptions(options);
        convert.setInputFormat(train);

        train = Filter.useFilter(train, convert);
        classifier.buildClassifier(train);
    }

    public void test() throws Exception {

        NumericToNominal convert = new NumericToNominal();
        String[] options = new String[2];
        options[0] = "-R";
        options[1] = "last";  //range of variables to make numeric


        evaluate.setClassIndex(evaluate.numAttributes() - 1);
        convert.setOptions(options);
        convert.setInputFormat(evaluate);

        evaluate = Filter.useFilter(evaluate, convert);
        classifier.buildClassifier(evaluate);
        evaluation = new Evaluation(evaluate);
        evaluation.evaluateModel(classifier, evaluate);


        System.out.println(classifier.toSummaryString());
        System.out.println(evaluation.toSummaryString("\nResults split\n======\n", false));
    }

    public void save_tree() throws Exception {

        PrintWriter out = new PrintWriter("tree.dot");
        out.print(classifier.graph());
        out.close();
        System.out.println("Tree saved");

    }
}