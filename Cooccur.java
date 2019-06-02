
import java.io.*;
import java.util.*;
import java.util.regex.Pattern;
class TermFreq implements Comparable<TermFreq> {

    int termId;
    int freq;

    TermFreq(int termId) {
        this.termId = termId;
    }

    public int compareTo(TermFreq that) {
        return Integer.compare(freq, that.freq);
    }
}

class Vocab {

    String fileName;
    int termId = 0;
    long collectionSize;
    HashMap<String, TermFreq> termIdMap;
    HashMap<Integer, String> idToStrMap;

    // Quick and dirty way to specify some more stopwords (if your preprocessor has missed some)
    static final String[] stopwords = {"br", "html", "http", "www", "htmlhttpwww", "linkshttpwww", "com]"};

    Vocab(String fileName) {
        this.fileName = fileName;
        termIdMap = new HashMap<>();
        idToStrMap = new HashMap<>();
        collectionSize = 0;
    }

    boolean isStopword(String word) {
        for (String stp : stopwords) {
            if (word.equals(stp)) {
                return true;
            }
        }
        return false;
    }



    void buildVocab() throws Exception {
        FileReader fr = new FileReader(fileName);
        BufferedReader br = new BufferedReader(fr);
        TermFreq tf = null;

        String line;

        while ((line = br.readLine()) != null) {
            String[] tokens = line.split("\\s+");

            for (String token : tokens) {
                if (isStopword(token)) {
                    continue;
                }

                if (!termIdMap.containsKey(token)) {
                    tf = new TermFreq(termId);
                    tf.freq++;
                    termIdMap.put(token, tf);
                    idToStrMap.put(new Integer(termId), token);
                    termId++;
                } else {
                    tf = termIdMap.get(token);
                    tf.freq++;  // collection freq
                }
            }
            collectionSize += tokens.length;
        }

        System.out.println(String.format("Initialized vocabulary comprising %d terms", termId));
        br.close();
        fr.close();
    }

    public HashSet<String> findUniqueWord() throws FileNotFoundException, IOException {
        FileReader fr = new FileReader(new File("/home/procheta/Node2Vec/npword2vec/npword2vec-master/datasets/men/MEN_dataset_natural_form_full"));
        BufferedReader br = new BufferedReader(fr);
        String line = br.readLine();
        
        HashSet<String> words = new HashSet<>();
        while (line != null) {
            String st[] = line.split(" ");
            try {
                words.add(st[0]);
                words.add(st[1]);
            } catch (Exception e) {
            }
            line = br.readLine();
        }
    
	return words;
	}


    void pruneVocab(float headp, float tailp)throws FileNotFoundException, IOException {
        System.out.println("Pruning vocabulary...");

        int size = vocabSize();
        int maxTf = 0;
        for (TermFreq tf : termIdMap.values()) {
            if (tf.freq > maxTf) {
                maxTf = tf.freq;
            }
        }

        int minCutOff = (int) (maxTf * headp);
        int maxCutOff = (int) (maxTf * tailp);

        System.out.println("Removing words with freq lower than " + minCutOff + " and higher than " + maxCutOff);
	Iterator<Map.Entry<String, TermFreq>> iter = termIdMap.entrySet().iterator();
        while (iter.hasNext()) {
            Map.Entry<String, TermFreq> entry = iter.next();
            TermFreq tf = entry.getValue();
            if ((tf.freq <= minCutOff || tf.freq >= maxCutOff) ) {
	       //if ((tf.freq <= minCutOff || tf.freq >= maxCutOff) ) {
   			//System.out.println(words.contains(getTerm(tf.termId)));
                        //System.out.println(getTerm(tf.termId));
	                iter.remove();
   
                idToStrMap.remove(tf.termId);
            }
        }

        System.out.println("vocab reduced to size " + termIdMap.size());
    }

    int getTermId(String word) {
        return termIdMap.containsKey(word) ? termIdMap.get(word).termId : -1;
    }

    TermFreq getTermFreq(String word) {
        return termIdMap.get(word);
    }

    String getTerm(int id) {
        return idToStrMap.get(id);
    }

    int vocabSize() {
        return termId;
    }
}

class CooccurStats {

    TermFreq a;
    TermFreq b;
    float p;  // conditional probability
    int count;
    int count_window;
    static final float ALPHA = 0.6f;
    static final float ONE_MINUS_ALPHA = 1.0f - ALPHA;

    CooccurStats(TermFreq a, TermFreq b) {
        this.a = a;
        this.b = b;
    }

    String encode(Vocab v) {
        String astr = v.getTerm(a.termId);
        String bstr = v.getTerm(b.termId);
        return String.format("%s\t%s\t%.4f", astr, bstr, p);
    }

    void normalize(Vocab v, float alpha) {

        TermFreq atf = v.getTermFreq(v.getTerm(a.termId));
        TermFreq btf = v.getTermFreq(v.getTerm(b.termId));
        int dfa = atf.freq;
        int dfb = btf.freq; // vocab gives coll freqs

        //p = ALPHA*p;
        p = alpha * p;
        //p = p + (ONE_MINUS_ALPHA) * (float)(Math.log(1 + (double)v.collectionSize/(double)(dfa + dfb)));
        p = p + (1 - alpha) * (float) (Math.log(1 + (double) v.collectionSize / (double) (dfa + dfb)));
    }
}

class CooccurMap {

    HashMap<String, CooccurStats> map;

    CooccurMap(Vocab v) {
        map = new HashMap<>((v.vocabSize() << 3));
    }

    void add(TermFreq tf1, TermFreq tf2, float delP,int c,int cw) {
        String key = tf1.termId + ":" + tf2.termId;
        CooccurStats seenStats = map.get(key);
        if (seenStats == null) {
            key = tf2.termId + ":" + tf1.termId;
            seenStats = map.get(key);
        }
        if (seenStats == null) {
            seenStats = new CooccurStats(tf1, tf2);
            map.put(key, seenStats);
        }
        seenStats.p += delP;
        seenStats.count += c;
        seenStats.count_window += cw;
    }

    int size() {
        return map.size();
    }
}

class DocTermMatrix {

    CooccurMap map;
    String fileName;
    Vocab v;
    HashMap<Integer, TermFreq> tfvec;
    TermFreq[] buff;
    int buffSize;
    Boolean posFlag;
    static final int MAX_DOC_SIZE = 22000;

    DocTermMatrix(String fileName, float headp, float tailp, Boolean posFlag) {
        this.fileName = fileName;

        try {
            v = new Vocab(fileName);
            v.buildVocab();
            v.pruneVocab(headp, tailp);
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        map = new CooccurMap(v);
        tfvec = new HashMap<>();
        buff = new TermFreq[MAX_DOC_SIZE];
        this.posFlag = posFlag;
    }

    void reinit() {
        buffSize = 0;
        tfvec.clear();
    }

    void vectorizePos(String[] tokens) {
        for (int i = 0; i < tokens.length; i++) {
            int termId = v.getTermId(tokens[i]);
            if (termId == -1) {
                continue;
            }
            TermFreq tf = new TermFreq(termId);
            tf.freq++;
	    buff[buffSize++] = tf;
        }
    }

    void vectorizeNoPos(String[] tokens) {
        for (int i = 0; i < tokens.length; i++) {
            int termId = v.getTermId(tokens[i]);
            if (termId == -1) {
                continue;
            }

            TermFreq seenTf = tfvec.get(termId);
            if (seenTf == null) {
                seenTf = new TermFreq(termId);
                tfvec.put(termId, seenTf);
            }
            seenTf.freq++;
        }

        for (TermFreq tf : tfvec.values()) {
            buff[buffSize++] = tf;
        }
    }

    void vectorize(String line) {
        reinit();

        String[] tokens = line.split("\\s+");
        if (!posFlag) {
            vectorizeNoPos(tokens);
        } else if (posFlag) {
            vectorizePos(tokens);
        }
    }

    void compute() throws Exception {
        FileReader fr = new FileReader(fileName);
        BufferedReader br = new BufferedReader(fr);

        String line;
        int termId;
        int docId = 0;
        float delP;

        while ((line = br.readLine()) != null) {

            vectorize(line);

            for (int i = 0; i < buffSize - 1; i++) {
                TermFreq a = buff[i];
                int c=0;
                int cw = 0;
                for (int j = i + 1; j < buffSize; j++) {  // inner loop is only for pairwise stats
                    TermFreq b = buff[j];

                    if (!posFlag) {
                        delP = (float) (Math.log(1 + a.freq / (float) buffSize) + Math.log(1 + b.freq / (float) buffSize));
                    } else {
                        //delP = (float) (Math.log(1 + a.freq / (float) buffSize) + Math.log(1 + b.freq / (float) buffSize)) * (float) Math.exp(-((j - i) * (j - i)));
                         delP =  (float) Math.exp(-((j - i) * (j - i)));
		                     if((j-i) > 3)
                           c = 1;
                          else
                            cw = 1;
										}
                    map.add(a, b, delP,c,cw);
                }
            }
            docId++;
            if (docId % 10000 == 0) {
                System.out.println("processed document " + docId);
            }
        }
        br.close();
        fr.close();
    }
}

class Cooccur {

    String fileName;
    DocTermMatrix dtmat;
    String outFile;
    float headp, tailp;

    Cooccur(String fileName, String outFile, float headp, float tailp) {
        this.fileName = fileName;
        this.outFile = outFile;
        this.headp = headp;
        this.tailp = tailp;
    }


    public HashMap<String, Double> getContextCooccurVal(String fileName) throws FileNotFoundException, IOException{
        FileReader fr = new FileReader(new File(fileName));
        BufferedReader br = new BufferedReader(fr);  
        String line = br.readLine();
        
        HashMap<String, Double> wordValMap = new HashMap<>();
        while(line != null){
            String st[] = line.split("\t");
	    			double d = .001;
	    			if(Pattern.matches("[a-zA-Z]+", st[0]) && Pattern.matches("[a-zA-Z]+", st[1]) &&  (Double.parseDouble(st[2]) > d) )	
	    				wordValMap.put(st[0]+"#"+st[1], Double.parseDouble(st[2]));        
						line = br.readLine();
				}
        return wordValMap;
    }



    public HashMap<String, Double> findMaxForEachWord(String fileName) throws FileNotFoundException, IOException {
        FileReader fr = new FileReader(new File(fileName));
        BufferedReader br = new BufferedReader(fr);

        String line = br.readLine();;
        HashMap<String, Double> wordMax = new HashMap<>();

        while (line != null) {
            String st[] = line.split("\t");
            if (wordMax.containsKey(st[0])) {
                if (Double.parseDouble(st[2]) > wordMax.get(st[0])) {
                    wordMax.put(st[0], Double.parseDouble(st[2]));
                }
            } else {
                wordMax.put(st[0], Double.parseDouble(st[2]));
            }

            if (wordMax.containsKey(st[1])) {
                if (Double.parseDouble(st[2]) > wordMax.get(st[1])) {
                    wordMax.put(st[1], Double.parseDouble(st[2]));
                }
            } else {
                wordMax.put(st[1], Double.parseDouble(st[2]));
            }
	    line = br.readLine();
        }
        return wordMax;
    }


    public void modifyCooccurGraph(double alpha, String fileName1, String fileName2, String fileName3) throws IOException {
        HashMap<String, Double> wordContexMap = getContextCooccurVal(fileName1);
        HashMap<String, Double> wordMax = findMaxForEachWord(fileName2);
        FileWriter fw = new FileWriter(new File(fileName3));
        BufferedWriter bw = new BufferedWriter(fw);

        FileReader fr = new FileReader(new File(fileName2));
        BufferedReader br = new BufferedReader(fr);

        String line = br.readLine();

        while (line != null) {
            String st[] = line.split("\t");
            String key = null;
            if (wordContexMap.containsKey(st[0] + "#" + st[1])) {
                key = st[0] + "#" + st[1];
            }
            if (wordContexMap.containsKey(st[1] + "#" + st[0])) {
                key = st[1] + "#" + st[0];
            }

            double max = 0;
            if (wordMax.get(st[1]) < wordMax.get(st[0])) {
                max = wordMax.get(st[0]);
            } else {
                max = wordMax.get(st[1]);
            }
            if((Double.parseDouble(st[2]) / max) > .001){
	    			if (key != null) {
                	bw.write(st[0] + "\t" + st[1] +"\t"+ String.valueOf(alpha * (Double.parseDouble(st[2]) / max)+(1 - alpha) * wordContexMap.get(key)));
                	bw.newLine();
                	wordContexMap.remove(key);
            	} else {
                	bw.write(st[0] + "\t" + st[1] +"\t"+String.valueOf((Double.parseDouble(st[2]) / max)));
                	bw.newLine();
            	}
            }
	     			line = br.readLine();
        }


				Iterator it = wordContexMap.keySet().iterator();

        System.out.println(wordContexMap.size());
        while (it.hasNext()) {
            String st = (String) it.next();
            String st1[] = st.split("#");
            bw.write(st1[0] + "\t" + st1[1] + "\t" + String.valueOf((wordContexMap.get(st))));
            bw.newLine();
        }
        bw.close();
    }    
    public void compute(float alpha, Boolean posFlag) {
        try {
            dtmat = new DocTermMatrix(fileName, headp, tailp, posFlag);
            dtmat.compute();

            FileWriter fw = new FileWriter(outFile);
            BufferedWriter bw = new BufferedWriter(fw);

            for (CooccurStats stats : dtmat.map.map.values()) {
                stats.normalize(dtmat.v, alpha);
                bw.write(stats.encode(dtmat.v));
                bw.newLine();
            }

            bw.close();
            fw.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public static void main(String[] args)throws FileNotFoundException, IOException {
        if (args.length < 6) {
            System.out.println("usage: java Cooccur <input text file (each line a document)> <output file path> <head %-le> <tail %-le> <alpha> <usePosition><UseContext><Context FilePath> <Final OutputFile Path><alpha>");
            return;
        }
        String inputFile = args[0];
        String oFile = args[1];
        float headp = Float.parseFloat(args[2]) / 100;
        float tailp = Float.parseFloat(args[3]) / 100;
        Boolean usePositions = Boolean.parseBoolean(args[5]);
				Cooccur c = new Cooccur(inputFile, oFile, headp, tailp);
        c.compute(Float.parseFloat(args[4]), usePositions);		
				if(args.length > 6)
				{
					Boolean useContext =	Boolean.parseBoolean(args[6]);
					String contextFilePath = args[7];
        	String finalOutputFile = args[8];
					double alpha = Double.parseDouble(args[9]);
					if(useContext){
          	c.modifyCooccurGraph(alpha,contextFilePath,oFile,finalOutputFile);
        	}
				}
   	}
}
