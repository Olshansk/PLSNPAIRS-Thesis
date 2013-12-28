package pls.othertools.rvptool;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import npairs.io.NpairsjIO;

import pls.shared.MLFuncs;

import com.jmatio.io.MatFileFilter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLStructure;

import extern.ArrayFuncs;
import extern.NewMatFileReader;
import pls.chrome.result.view.PredictionPlot;
/**
 * Holds calculated means and medians for each prediction by pc # group.
 * Each group has a given arbitrary name and a selected session/subject.
 * For each group the mean and median value are extracted out of the specified
 * result files for the selected subject and then combined to give a collection
 * of means and medians; one of each for each result file.
 * Prediction values used are either posterior probability or percent accuracy.
 */
class PPlotData {

	private ArrayList<CurveGroup> curves;
	private HashSet<String> filesToLoad = new HashSet<String>();
	private String errorString = "";
	private boolean mPlotPercentAccuracy = false; // if true, use prediction percent accuracy instead
	                                              // of posterior probabilities when calculating means.

	//arraylist index denotes split object.
	private Map<String,ArrayList<Double>> fileMeans
						= new HashMap<String,ArrayList<Double>>(200);
	private Map<String,ArrayList<Double>> fileMedians
						= new HashMap<String, ArrayList<Double>>(200);
	private Map<String,Set<CurveGroup>> fileMap
						= new HashMap<String, Set<CurveGroup>>(200);
	private Map<String,String> splitObjType = new HashMap<String,String>(200);
	
	/** Constructor.  Extracts means from input NPAIRS result files.  M
	 * 
	 * @param setupInfo input files
	 * @param plotPercentAccuracy If true, calculate means of prediction percent accuracy values
	 * instead of posterior probability values.
	 */
	public PPlotData(ArrayList<CurveGroup> setupInfo, boolean plotPercentAccuracy){
		curves = setupInfo;
		discoverFiles(setupInfo);
		mPlotPercentAccuracy = plotPercentAccuracy;
		for(String filename : filesToLoad){
			openFileCalcMeans(filename); // extract means from files.
		}
	}

	ArrayList<CurveGroup> getCurves(){
		return curves;
	}
	/**
	 * Determine what result files are needed for the plot.
	 * @param setupInfo Group information detailing which result files belong
	 * to which group.
	 */
	private void discoverFiles(ArrayList<CurveGroup> setupInfo){
		// Removes duplicate files so we don't load them twice.
		for(CurveGroup rg : setupInfo){
			for(String num : rg.getParsedNumbers()){
				String file = rg.getFilename().replace("$", num);
				filesToLoad.add(file);
				
				Set<CurveGroup> gSet;
				if(!fileMap.containsKey(file)){
					gSet = new HashSet<CurveGroup>();
					fileMap.put(file, gSet);
				}
				gSet = fileMap.get(file);
				gSet.add(rg);
			}
		}
	}

	/**
	 * Check to see if any group specifies a subject that is contained in this
	 * file. If at least one group uses this file correctly (uses a 
	 * subject that exists) then this file should be loaded. A warning for 
	 * groups that use this file incorrectly will be issued later in PPlot
	 * in generateDataset(). If no group specifies a subject to use in this 
	 * file then do not waste time loading this file.
	 * @param filename the result file we currently attempting to load.
	 * @param subjLimit the number of subjects that this result file contains.
	 * @return true if this file is used by at least one group correctly, false
	 * otherwise.
	 */
	private boolean subjectsValid(String filename, int subjLimit){
		Set<CurveGroup> gSet = fileMap.get(filename);
		
		for(CurveGroup group : gSet){
			if (group.getCurveUnit() <= subjLimit) return true;
		}
		return false;
	}
	
	/**
	 * Open and calculate the session/subject means for the given file.
	 * @param filename the file to load means for.
	 */
	private void openFileCalcMeans(String filename){
		Map<String, MLArray> resultInfo = null;
		MLStructure struct_npairs_result;
		MLStructure struct_prediction;
		MLStructure struct_priors;
		MLArray temp;

		double[][] ppTrueClass = null;
		double[][] correctPred = null; 
		double[] subjLabels = null;
		double[][] splitVols = null;

		String field = "split_obj_labels";
		
		try {
			 resultInfo = new NewMatFileReader(filename,
					 new MatFileFilter(new String[]{"npairs_result"}))
					 .getContent();

			 MLArray npairs_result = resultInfo.get("npairs_result");
			 if (npairs_result != null && npairs_result.isStruct()) {
				struct_npairs_result = (MLStructure) npairs_result;

				temp = struct_npairs_result.getField(field);
				if(temp == null){
					// try loading using obsolete variable name instead
					field = "subj_labels";
					temp = struct_npairs_result.getField("subj_labels");
				}
				
				if(temp != null){
					subjLabels = MLFuncs.getRow(((MLDouble) temp).getArray(), 0);
					int [] realSubjLabel;
					realSubjLabel = (int []) ArrayFuncs.convertArray(subjLabels,
																	int.class);
					if(!subjectsValid(filename,
							MLFuncs.unique(realSubjLabel).length)){
						/*This file does not contain subjects used by any of
						the curves/groups and is thus not loaded.*/
						errorString += "\nThe file " + filename + "\nhas not been"
						+ " loaded because no curves have specified valid " +
						"subjects to use.\n";
						return;
					}
				}else{
					errorString += "Missing required 'split_obj_labels'" +
							"/'subj_labels' " +	"field for file: " 
							+ filename + "\n";
					return;
				}
				
				temp = struct_npairs_result.getField("split_type");
				if(temp != null){
					String splitType = ((MLChar) temp).getString(0);
					splitObjType.put(filename, splitType);
				}else{
					splitObjType.put(filename, "Split obj");
				}

				temp = struct_npairs_result.getField("split_test_vols");
				if(temp != null){
					splitVols = ((MLDouble) temp).getArray();
				}else{
					errorString += "The field 'split_test_vols' was unreadable"
					+ " for file: " + filename;
					return;
				}

				temp = struct_npairs_result.getField("prediction");
				if (temp != null && temp.isStruct()) {
					struct_prediction = (MLStructure) temp;

					temp = struct_prediction.getField("priors");
					if (temp != null && temp.isStruct()) {
						struct_priors = (MLStructure) temp;

						if (mPlotPercentAccuracy) {
							temp = struct_priors.getField("correct_pred");
							if (temp != null) {
								correctPred = (((MLDouble) temp).getArray());
							} else {
								errorString += "The field 'correct_pred' was " +
									"unreadable for file: " + filename;
								return;
							}
						} else {
							temp = struct_priors.getField("pp_true_class");
							if (temp != null) {
								ppTrueClass = (((MLDouble) temp).getArray());
							}else{
								errorString += "The field 'pp_true_class' was " +
								"unreadable for file: " + filename;
								return;
							}
						}
					}
					else{
						errorString += "The field 'priors' was unreadable " +
						"for file: " + filename;
						return;
					}
				}
				else{
					errorString += "The field 'prediction' was unreadable " +
					"for file: " + filename;
					return;
				}
			 }else{
				 errorString += "The field 'npairs_result' was unreadable " +
				 		"for file: " + filename;
				 return;
			 }
		} catch (IOException e) {
			errorString += "Could not load " + filename + "\n";
			return;
		}

		ArrayList<ArrayList<Double>> subjMeans;
		ArrayList<Double> means, medians;

		/*All means for all splitvols (subjects) are calculated here
		  not just the ones used by the plot. later we just select the
		  means that are needed on a per subject basis*/
		if (mPlotPercentAccuracy) {
			subjMeans = calculateSubjectMeans(correctPred, splitVols, subjLabels);
		}
		else {
			subjMeans = calculateSubjectMeans(ppTrueClass,splitVols,subjLabels);
		}
		means = meanOfMeans(subjMeans);
		medians = medianOfMeans(subjMeans);
		fileMeans.put(filename,means);
		fileMedians.put(filename,medians);

	}

	/**
	 * See: PredictionPlot.getBySubjectDataset() for identical function.
	 * Calculate the subject/session means for a given file.
	 * @param mPPTrue
	 * @param splitVols
	 * @param splitLabels
	 * @return An arraylist of subjects (arraylists themselves) each containing
	 * possibly multiple means.
	 */
	private ArrayList<ArrayList<Double>> calculateSubjectMeans(double[][] mPPTrue,
			                           double[][] splitVols,
									   double[] splitLabels){

		ArrayList<ArrayList<Double>> vals = PredictionPlot.getBlankValuesArray(
				splitLabels);
		
		int numSubj = vals.size();
		double[] rowMean = null;
		int[] numAdded = null;

		for (int i = 0; i < splitVols.length; i++) {
			rowMean = new double[numSubj];
			numAdded = new int[numSubj];
			for (int j = 0; j < splitVols[i].length; j++) {
				int vol = (int) splitVols[i][j];
				double data = (double) mPPTrue[i][j];
				int subj = 0;
				if (vol > 0 && data >= 0) {
					subj = (int) splitLabels[vol - 1];
					rowMean[subj - 1] += data;
					numAdded[subj - 1]++;
				}
			}

			//integer p+1 denotes the actual subject. so if p = 5 then
			//vals.get(p) represents the row means for subject 6.
			for (int p = 0; p < numSubj; p++) {
				if(numAdded[p] != 0){
					rowMean[p] /= numAdded[p];
					vals.get(p).add(rowMean[p]);
				}
			}
		}
		return vals;
	}

	/**
	 * See: PredictionPlot.getSubjectMeans() for identical function
	 * For each subject calculate the mean of their means.
	 * @return an arraylist containing a single mean value for each subject.
	 */
	private ArrayList<Double> meanOfMeans(
			ArrayList<ArrayList<Double>> subjMeans){
		ArrayList<ArrayList<Double>> vals = subjMeans;
		ArrayList<Double> subjectMeans = new ArrayList<Double>(vals.size());
		Double temp = 0.0;
		//Calculate each subject's mean.
		for (ArrayList<Double> subject : vals){
			if(subject.size() == 0){
				subjectMeans.add(-1.0); //indicator that subject has no means.
			}
			else{
				for (Double val : subject) {
					temp = temp + val;
				}
				subjectMeans.add(temp/subject.size());
				temp = 0.0;
			}
		}
		return subjectMeans;
	}

	/**
	 * See: PredictionPlot.getSubjectMedians() for identical function
	 * same as meanOfMeans() but this time we are looking at medians.
	 * @return the median of the means for each subject.
	 */
	private ArrayList<Double> medianOfMeans(
			ArrayList<ArrayList<Double>> subjMeans){
		ArrayList<ArrayList<Double>> vals = subjMeans;
		ArrayList<Double> subjectMedian = new ArrayList<Double>(vals.size());

		for (ArrayList<Double> subject : vals){
			if(subject.size() == 0){
				subjectMedian.add(-1.0);
			}
			else{

				Double[] sortedVals = new Double[subject.size()];
				subject.toArray(sortedVals);
				Arrays.sort(sortedVals);

				int middleIndex = sortedVals.length / 2;

				if(sortedVals.length % 2 == 0){
					subjectMedian.add(
							(sortedVals[middleIndex]
                            + sortedVals[middleIndex - 1]) / 2);
				}else{
					subjectMedian.add(sortedVals[middleIndex]);
				}

			}
		}
		return subjectMedian;
	}

	public Map<String, ArrayList<Double>> getFileMeans() {
		return fileMeans;
	}

	public Map<String, ArrayList<Double>> getFileMedians() {
		return fileMedians;
	}
	
	public Map<String,String> getSplitObjType(){
		return splitObjType;
	}

	/**
	 *
	 * @return any error messages generated while loading result files.
	 */
	public String getErrorString(){
		return errorString;
	}
	
	public static void main(String args[]){
		testLoader();
	}
	/**
	 * Test that multiple result files can be loaded. Once loaded test that
	 * session information can be extracted from specific subsets of the loaded
	 * result files as specified in group information.
	 */
	private static void testLoader(){
		boolean error = false;

		ArrayList<CurveGroup> setupInfo = new ArrayList<CurveGroup>();

		//create three groups and test that session means/medians can be loaded
		//for session numbers #1,2,3 across result files 2,6,10.
		setupInfo.add(new CurveGroup("G1",
		"/home/blessburn/work/PLSwithNPAIRS/data/ATOL2/jul30_2010_y246_5cond_$pc_NPAIRSJresult.mat",
				1,"2,6,10"));
		setupInfo.add(new CurveGroup("G2",
		"/home/blessburn/work/PLSwithNPAIRS/data/ATOL2/jul30_2010_y246_5cond_$pc_NPAIRSJresult.mat",
				2,"2,6,10"));
		setupInfo.add(new CurveGroup("G3",
		"/home/blessburn/work/PLSwithNPAIRS/data/ATOL2/jul30_2010_y246_5cond_$pc_NPAIRSJresult.mat",
				3,"2,6,10"));

		double[] pc002Medians = {0.20668132549950055, //session #1
			                   0.23220241140751613, //session #2
							   0.24163717835668533}; //session #3
		double[] pc002Means = {0.2104472959286006,
		                         0.22919874443184762,
								 0.24102653965117438};

		double[] pc006Medians = {0.29645976220952375,	0.2899131596029278,0.3141370610821815};

		double[] pc006Means = {0.29915097531783136,0.29925919804973383,0.3197849735472273};

		double[] pc010Medians = {0.3237788455509063,0.36009056529858047,0.3372683160841772};
		double[] pc010Means = {0.3306912343300008,0.3810720280222018,0.3552376977392077};

		boolean plotPercentAccuracy = true;
		PPlotData p = new PPlotData(setupInfo, plotPercentAccuracy);

		for(CurveGroup group : setupInfo){
			for(String num : group.getParsedNumbers()){
				String file = group.getFilename().replace("$", num);
				int session = group.getCurveUnit();
				double mean = ((ArrayList<Double>) p.fileMeans.get(file)).get(session-1);
				double median = ((ArrayList<Double>) p.fileMedians.get(file)).get(session-1);
				if(num.equals("002")){
					if(pc002Means[session-1] != mean || pc002Medians[session-1] != median){
						error = true;
						System.err.println("Means: Want -> " + pc002Means[session-1] +
								" Have -> " + mean);
						System.err.println("Medians: Want -> " + pc002Medians[session-1] +
								" Have -> " + median);
					}
				}
				if(num.equals("006")){
					if(pc006Means[session-1] != mean || pc006Medians[session-1] != median){
						error = true;
						System.err.println("Means: Want -> " + pc002Means[session-1] +
								" Have -> " + mean);
						System.err.println("Medians: Want -> " + pc002Medians[session-1] +
								" Have -> " + median);
					}
				}
				if(num.equals("010")){
					if(pc010Means[session-1] != mean || pc010Medians[session-1] != median){
						error = true;
						System.err.println("Means: Want -> " + pc002Means[session-1] +
								" Have -> " + mean);
						System.err.println("Medians: Want -> " + pc002Medians[session-1] +
								" Have -> " + median);
					}
				}
			}
			System.out.println("Processing curve " + group.getLabel());
		}
		if(error){
					System.err.println("Error in pdata calculations");
		} else {
			System.err.println("All tests pass");
		}
	}
	
}
