package plagiarismdetection;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class SAEvaluate {

	public class SA {
		String id = "", text = "", tag = "";
		List<String> aspects = new ArrayList<String>();
		List<String> values = new ArrayList<String>();
	}

	public List<SA> loadSA(String filename) {
		List<String> inputString = getInputFromFile(filename);
		List<SA> sa = new ArrayList<SA>();
		SA senti = null;
		int i = 0;

		while (i < inputString.size()) {
			String line = inputString.get(i);
			line = line.replace("#", "#");
			line = line.trim();

			if (line.matches(".?#\\d+")) {
				if (senti != null) sa.add(senti);
				senti = new SA();
				senti.id = line;
			} else {
				if (line.matches("\\{([A-Za-z]+)(.*)([a-z]+)\\}")) {
					senti.tag = line;
					String[] tokens = line.split(",");

					for (int t = 0; t < tokens.length; t = t + 2) {
						tokens[t] = tokens[t].replace("{", "");
						tokens[t] = tokens[t].replace("}", "");
						tokens[t] = tokens[t].trim();
						senti.aspects.add(tokens[t]);
					}

					for (int t = 1; t < tokens.length; t = t + 2) {
						tokens[t] = tokens[t].replace("{", "");
						tokens[t] = tokens[t].replace("}", "");
						tokens[t] = tokens[t].trim();
						senti.values.add(tokens[t]);
					}

				} else if (!line.equals("") && line.length() > 10) {
					System.out.println(line);
					senti.text = line;
				}
			}
			i++;
		}
		sa.add(senti);
		return sa;
	}

	private boolean checkDuplicateAsp(String asp, List<String> aspList) {
		int count = 0;
		for (int i = 0; i < aspList.size(); i++)
			if (aspList.get(i).equals(asp)) count++;
		return count > 1;
	}

	public void evaluate(String gold, String answer) {
		List<String> allAspects = new ArrayList<String>();
		List<Integer> goldAspCount = new ArrayList<Integer>();

		List<Integer> ansAllAspCount = new ArrayList<Integer>();
		List<Integer> ansAspCount = new ArrayList<Integer>();
		List<Integer> ansValueCount = new ArrayList<Integer>();

		List<SA> goldSA = loadSA(gold);
		List<SA> ansSA = loadSA(answer);

		for (int i = 0; i < goldSA.size(); i++) {
			List<String> asp = goldSA.get(i).aspects;
			for (int j = 0; j < asp.size(); j++)
				if (!allAspects.contains(asp.get(j))) {
					allAspects.add(asp.get(j));
					System.out.println(asp.get(j));
					System.out.println(goldSA.get(i).text);
				}
		}
		System.out.println(allAspects);

		for (int i = 0; i < allAspects.size(); i++) {
			goldAspCount.add(0);
			ansAspCount.add(0);
			ansValueCount.add(0);
			ansAllAspCount.add(0);
		}

		for (int i = 0; i < goldSA.size(); i++) {
			List<String> asp = goldSA.get(i).aspects;
			for (int j = 0; j < asp.size(); j++) {
				int id = allAspects.indexOf(asp.get(j));
				if (id != -1) goldAspCount.set(id, goldAspCount.get(id) + 1);
				else System.out.println("!!! ERROR " + goldSA.get(i).id);
			}
		}

		for (int i = 0; i < ansSA.size(); i++) {
			List<String> asp = ansSA.get(i).aspects;
			for (int j = 0; j < asp.size(); j++) {
				int id = allAspects.indexOf(asp.get(j));
				if (id != -1) ansAllAspCount.set(id, ansAllAspCount.get(id) + 1);
				else System.out.println("!!! Warning " + asp.get(j));
			}
		}

		for (int i = 0; i < goldSA.size(); i++) {
			SA g = goldSA.get(i);
			SA a = ansSA.get(i);

			if (!g.id.equals(a.id)) System.out.println("Lỗi gióng hàng:" + g.id + " <-> " + a.id);
			else {
				if (g.text.compareTo(a.text) != 0) System.out.println("Lỗi văn bản:" + a.id + "\n[" + g.text + "]\n<-> \n[" + a.text + "]" );
				List<String> gasp = g.aspects;
				List<String> aasp = a.aspects;
				List<String> gval = g.values;
				List<String> aval = a.values;

				for (int t = 0; t < gasp.size(); t++) {
					int id = aasp.indexOf(gasp.get(t));
					if (id != -1) {
						String aspect = gasp.get(t);
						int gid = allAspects.indexOf(aspect);
						ansAspCount.set(gid, ansAspCount.get(gid) + 1);

						if (!checkDuplicateAsp(aspect, aasp))
							if (gval.get(t).equals(aval.get(id))) ansValueCount.set(gid, ansValueCount.get(gid) + 1);
						else System.out.println(a.text);
					}
				}
			}
		}

		System.out.println("Evaluation Result >> File:" + answer + "<> [" + gold + "]");
		System.out.printf("%30s", " ");
		for (int i = 0; i < allAspects.size(); i++) System.out.printf("\t%s", "asp#" + (i + 1));
		System.out.println();

		System.out.printf("%30s", "Gold count");
		for (int i = 0; i < allAspects.size(); i++) System.out.printf("\t%d", goldAspCount.get(i));
		System.out.println();

		System.out.printf("%30s", "ANSWER count");
		for (int i = 0; i < allAspects.size(); i++) System.out.printf("\t%d", ansAllAspCount.get(i));
		System.out.println();
		System.out.println();

		System.out.printf("%30s", "Correct ANSWER: aspect");
		for (int i = 0; i < allAspects.size(); i++) System.out.printf("\t%d", ansAspCount.get(i));
		System.out.println();

		System.out.printf("%30s", "Precision: aspect");
		for (int i = 0; i < allAspects.size(); i++) {
			double p = 0;
			if (ansAllAspCount.get(i) > 0) p = 1.0 * ansAspCount.get(i) / ansAllAspCount.get(i);
			System.out.printf("\t%.4f", p);
		}
		System.out.println();

		System.out.printf("%30s", "Recall: aspect");
		for (int i = 0; i < allAspects.size(); i++) {
			double r = 0;
			if (goldAspCount.get(i) > 0) r = 1.0 * ansAspCount.get(i) / goldAspCount.get(i);
			System.out.printf("\t%.4f", r);
		}
		System.out.println();

		System.out.printf("%30s", "F1 score: aspect");
		for (int i = 0; i < allAspects.size(); i++) {
			double p = 0, r = 0, f = 0;
			if (ansAllAspCount.get(i) > 0) p = 1.0 * ansAspCount.get(i) / ansAllAspCount.get(i);
			if (ansAllAspCount.get(i) > 0) r = 1.0 * ansAspCount.get(i) / goldAspCount.get(i);
			if (p + r > 0) f = 2.0 * (p * r) / (p + r);
			System.out.printf("\t%.4f", f);
		}

		int tgold = 0;
		for (int i = 0; i < goldAspCount.size(); i++)
			tgold = tgold + goldAspCount.get(i);

		int tans = 0;
		for (int i = 0; i < ansAllAspCount.size(); i++)
			tans = tans + ansAllAspCount.get(i);

		int tcans = 0;
		for (int i = 0; i < ansAspCount.size(); i++)
			tcans = tcans + ansAspCount.get(i);

		int tvalue = 0;
		for (int i = 0; i < ansValueCount.size(); i++)
			tvalue = tvalue + ansValueCount.get(i);

		{
			double p = 1.0 * tcans / tans;
			double r = 1.0 * tcans / tgold;
			double f1 = 2 * p * r / (p + r);

			System.out.println();
			System.out.printf("%30s", "Over All ANSWER: aspect:----");
			System.out.printf("\t%.4f", p);
			System.out.printf("\t%.4f", r);
			System.out.printf("\t%.4f", f1);
			System.out.println();
			System.out.println();
		}

		System.out.printf("%30s", "Correct ANSWER: aspect,value");
		for (int i = 0; i < allAspects.size(); i++)
			System.out.printf("\t%d", ansValueCount.get(i));
		System.out.println();

		System.out.printf("%30s", "Precision: aspect, value");
		for (int i = 0; i < allAspects.size(); i++) {
			double p = 0;
			if (ansAllAspCount.get(i) > 0)
				p = 1.0 * ansValueCount.get(i) / ansAllAspCount.get(i);
			System.out.printf("\t%.4f", p);
		}
		System.out.println();

		System.out.printf("%30s", "Recall: aspect, value");
		for (int i = 0; i < allAspects.size(); i++) {
			double r = 0;
			if (ansAllAspCount.get(i) > 0)
				r = 1.0 * ansValueCount.get(i) / goldAspCount.get(i);
			System.out.printf("\t%.4f", r);
		}
		System.out.println();

		System.out.printf("%30s", "F1 score: aspect, value");
		for (int i = 0; i < allAspects.size(); i++) {
			double p = 0, r = 0, f = 0;
			if (ansAllAspCount.get(i) > 0) p = 1.0 * ansValueCount.get(i) / ansAllAspCount.get(i);
			if (goldAspCount.get(i) > 0) r = 1.0 * ansValueCount.get(i) / goldAspCount.get(i);
			if (p + r > 0) f = 2.0 * (p * r) / (p + r);
			System.out.printf("\t%.4f", f);
		}
		System.out.println();

		{
			double p = 1.0 * tvalue / tans;
			double r = 1.0 * tvalue / tgold;
			double f1 = 2 * p * r / (p + r);

			System.out.println();
			System.out.printf("%30s", "Over All ANSWER: aspect, value:----");
			System.out.printf("\t%.4f", p);
			System.out.printf("\t%.4f", r);
			System.out.printf("\t%.4f", f1);
			System.out.println();
			System.out.println();
		}
		System.out.println();
		for (int i = 0; i < allAspects.size(); i++)
			System.out.println("asp#" + (i + 1) + ": " + allAspects.get(i));

	}

	public void evaluateFolder(String gold, String answer) {
		File goldFolder = new File(gold);

		File ansFolder = new File(answer);
		String[] goldfiles = goldFolder.list();
		String[] ansfiles = ansFolder.list();

		for (int i = 0; i < ansfiles.length; i++) {
			String af = ansfiles[i].toLowerCase();
			if (af.contains("hotel")) evaluate(gold + "/" + goldfiles[0], answer + "/" + ansfiles[i]);
			else if (af.contains("restaurant")) evaluate(gold + "/" + goldfiles[1], answer + "/" + ansfiles[i]);
		}
	}

	private List<String> getInputFromFile(String input) {
		List<String> lines = new ArrayList<String>();
		try {
			Scanner reader = new Scanner(new FileReader(input));
			while (reader.hasNext()) {
				String line = reader.nextLine();
				// if(line.compareTo("")!=0)
				lines.add(line);
			}
			reader.close();
		} catch (FileNotFoundException e) { e.printStackTrace(); }
		return lines;
	}

	public static void main(String[] args) {
		SAEvaluate ltc = new SAEvaluate();
		String path = "C:\\Users\\Administrator\\Downloads\\";
		String answer3 = path + "MonoBERT_en_5.txt"; // Gold dataset
		String gold1 = "C:\\Users\\Administrator\\Downloads\\VLSP Formated SemEval\\En_Test.txt";
		ltc.evaluate(gold1, answer3);
	}
}