import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.io.IOException;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SAEvaluate {

    static class SA {
        String id = "", text = "", tag = "";
        List<String> aspects = new ArrayList<>();
        List<String> values = new ArrayList<>();
    }

    public List<SA> loadSA(String filename) throws IOException {
        List<SA> saList = new ArrayList<>();
        SA sa = null;

        for (String line : Files.readAllLines(Paths.get(filename))) {
            line = line.replace("#", "#").trim();

            if (line.matches(".?#\\d+")) {
                if (sa != null) saList.add(sa);
                sa = new SA();
                sa.id = line;
            } else if (line.matches("\\{([A-Za-z]+)(.*)([a-z]+)\\}")) {
                sa.tag = line;
                String[] tokens = line.replaceAll("[{}]", "").split(",");
                for (int i = 0; i < tokens.length; i += 2) {
                    sa.aspects.add(tokens[i].trim());
                    sa.values.add(tokens[i + 1].trim());
                }
            } else if (!line.isEmpty() && line.length() > 10) sa.text = line;
        }
        saList.add(sa);
        return saList;
    }

    public void evaluate(String goldFile, String answerFile) throws IOException {
        List<String> allAspects = new ArrayList<>();
        List<SA> goldSA = loadSA(goldFile), ansSA = loadSA(answerFile);
        Map<String, Integer> goldAspectCount = new HashMap<>();
        Map<String, Integer> ansAllAspectCount = new HashMap<>();
        Map<String, Integer> ansAspectCount = new HashMap<>();
        Map<String, Integer> ansValueCount = new HashMap<>();

        for (SA sa : goldSA) {
            for (String aspect : sa.aspects) {
                if (!allAspects.contains(aspect)) {
                    allAspects.add(aspect);
                    goldAspectCount.put(aspect, 0);
                    ansAllAspectCount.put(aspect, 0);
                    ansAspectCount.put(aspect, 0);
                    ansValueCount.put(aspect, 0);
                }
                goldAspectCount.put(aspect, goldAspectCount.get(aspect) + 1);
            }
        }

        for (SA sa : ansSA)
            for (String aspect : sa.aspects) 
                if (allAspects.contains(aspect)) ansAllAspectCount.put(aspect, ansAllAspectCount.get(aspect) + 1);
                else System.out.println("!!! Warning " + aspect);

        for (int i = 0; i < goldSA.size(); i++) {
            SA g = goldSA.get(i), a = ansSA.get(i);

            if (!g.id.equals(a.id)) System.out.println("Row mismatch:" + g.id + " <-> " + a.id);
            else if (!g.text.equals(a.text)) System.out.println("Text mismatch:" + a.id + "\n[" + g.text + "]\n<-> \n[" + a.text + "]");
            else {
                for (int j = 0; j < g.aspects.size(); j++) {
                    String gaspect = g.aspects.get(j);
                    int id = a.aspects.indexOf(gaspect);

                    if (id != -1) {
                        ansAspectCount.put(gaspect, ansAspectCount.get(gaspect) + 1);
                        if (a.aspects.indexOf(gaspect) == id && g.values.get(j).equals(a.values.get(id)))
                            ansValueCount.put(gaspect, ansValueCount.get(gaspect) + 1);
                    }
                }
            }
        }

        System.out.println("Evaluation Result >> File:" + answerFile + "<> [" + goldFile + "]");
        printEvaluation("Gold count", goldAspectCount, allAspects);
        printEvaluation("ANSWER count", ansAllAspectCount, allAspects);
        System.out.println();

        printEvaluation("Correct ANSWER: aspect", ansAspectCount, allAspects);
        printMetric("Precision: aspect", ansAspectCount, ansAllAspectCount, allAspects);
        printMetric("Recall: aspect", ansAspectCount, goldAspectCount, allAspects);
        printF1("F1 score: aspect", ansAspectCount, goldAspectCount, ansAllAspectCount, allAspects);

        int totalGold = goldAspectCount.values().stream().mapToInt(Integer::intValue).sum();
        int totalAns = ansAllAspectCount.values().stream().mapToInt(Integer::intValue).sum();
        int totalCorrectAns = ansAspectCount.values().stream().mapToInt(Integer::intValue).sum();
        printOverall("Over All ANSWER: aspect:----", totalCorrectAns, totalAns, totalGold);

        printEvaluation("Correct ANSWER: aspect,value", ansValueCount, allAspects);
        printMetric("Precision: aspect, value", ansValueCount, ansAllAspectCount, allAspects);
        printMetric("Recall: aspect, value", ansValueCount, goldAspectCount, allAspects);
        printF1("F1 score: aspect, value", ansValueCount, goldAspectCount, ansAllAspectCount, allAspects);

        int totalValue = ansValueCount.values().stream().mapToInt(Integer::intValue).sum();
        printOverall("Over All ANSWER: aspect, value:----", totalValue, totalAns, totalGold);

        System.out.println();
        for (int i = 0; i < allAspects.size(); i++)
            System.out.println("asp#" + (i + 1) + ": " + allAspects.get(i));
    }

    private void printEvaluation(String label, Map<String, Integer> data, List<String> allAspects) {
        System.out.printf("%30s", label);
        for (String aspect : allAspects) System.out.printf("\t%d", data.get(aspect));
        System.out.println();
    }

    private void printMetric(String label, Map<String, Integer> correct, Map<String, Integer> total, List<String> allAspects) {
        System.out.printf("%30s", label);
        for (String aspect : allAspects) {
            double p = total.get(aspect) > 0 ? 1.0 * correct.get(aspect) / total.get(aspect) : 0;
            System.out.printf("\t%.4f", p);
        }
        System.out.println();
    }

    private void printF1(String label, Map<String, Integer> correct, Map<String, Integer> gold, Map<String, Integer> total, List<String> allAspects) {
        System.out.printf("%30s", label);
        for (String aspect : allAspects) {
            double p = total.get(aspect) > 0 ? 1.0 * correct.get(aspect) / total.get(aspect) : 0;
            double r = gold.get(aspect) > 0 ? 1.0 * correct.get(aspect) / gold.get(aspect) : 0;
            double f = p + r > 0 ? 2.0 * (p * r) / (p + r) : 0;
            System.out.printf("\t%.4f", f);
        }
        System.out.println();
    }

    private void printOverall(String label, int totalCorrect, int totalPredicted, int totalGold) {
        double p = totalPredicted > 0 ? 1.0 * totalCorrect / totalPredicted : 0;
        double r = totalGold > 0 ? 1.0 * totalCorrect / totalGold : 0;
        double f1 = p + r > 0 ? 2 * p * r / (p + r) : 0;

        System.out.println();
        System.out.printf("%30s", label);
        System.out.printf("\t%.4f", p);
        System.out.printf("\t%.4f", r);
        System.out.printf("\t%.4f", f1);
        System.out.println();
        System.out.println();
    }

    public static void main(String[] args) throws IOException {
        SAEvaluate evaluator = new SAEvaluate();
        // String trueFilePath = "D:\\absa-vlsp-2018\\datasets\\vlsp2018_hotel\\3-VLSP2018-SA-Hotel-test.txt";
        // String predFilePath = "D:\\absa-vlsp-2018\\experiments\\predictions\\ACSA-v1-hotel.txt";
        evaluator.evaluate(args[0], args[1]);
    }
}