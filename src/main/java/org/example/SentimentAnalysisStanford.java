package org.example;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;

import java.util.Properties;
import java.util.Scanner;

public class SentimentAnalysisStanford {
    static StanfordCoreNLP pipeline;

    public static void main(String[] args) {
        init();
        Scanner sc = new Scanner(System.in);
        while (true) {
            System.out.println(findSentiment(sc.nextLine()));
        }
    }

    public static void init() {
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, pos, parse, sentiment");
        pipeline = new StanfordCoreNLP(props);
    }

    public static String findSentiment(String input) {
        String sentiment = "Neutral";
        if (input != null && input.length() > 0) {
            Annotation annotation = pipeline.process(input);
            for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
                Tree tree = sentence.get(SentimentCoreAnnotations.SentimentAnnotatedTree.class);
                int sentimentScore = RNNCoreAnnotations.getPredictedClass(tree);
                switch (sentimentScore) {
                    case 0, 1:
                        sentiment = "Negative";
                        break;
                    case 3, 4:
                        sentiment = "Positive";
                        break;
                    default:
                        sentiment = "Neutral";
                        break;
                }
            }
        }
        return sentiment;
    }
}
