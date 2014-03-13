package npairs;

import npairs.io.NpairsjIO;

import java.io.*;
import java.util.*;

import java.util.Vector;

//import javax.swing.JOptionPane;

import com.jmatio.io.MatFileFilter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLStructure;

import extern.NewMatFileReader;
//import extern.ArrayFuncs;

import pls.chrome.sessionprofile.*;
import pls.sessionprofile.RunInformation;
import pls.shared.BfMRISessionFileFilter;
import pls.shared.MLFuncs;
import pls.shared.NpairsfMRIResultFileFilter;
import pls.shared.StreamedProgressHelper;

import java.util.Date;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.text.DecimalFormat;
import extern.niftijlib.Nifti1Dataset;

/**
 * This class is essentially a structure containing all the parameters used to
 * run an NPAIRSJ analysis. 
 * 
 * @author anita oder
 * 
 */
public class NpairsjSetupParams implements Serializable{

	public boolean useQuadCVA = false; // Added by Anita Apr 27 2010.
	
	final boolean debug = false;

	/** Contains session info loaded from each session file;
	 * length: no. sess files
	 * 
	 */
	transient private NpairsSessionInfo[] sessInfo; 

	/** Which runs to include when loading session files. If this variable 
	 *  is not set, inclRuns info will be loaded from session files. Otherwise
	 *  this variable will override whatever inclRuns info may be saved in session
	 *  files.
	 */
	public int[] inclRuns = null;
	
	private int nDataFiles; 

	private int nSkipTmpts;

	// **Data loading**

	/** Array indicating whether each input condition (class) label is to
	 *  be included or excluded from current analysis.  Used only when
	 *  data is loaded from datamats.  Given condition labels
	 *  {1,2,3,4,5}, e.g., conditionSelection = {1,0,1,0,1} means include
	 *  only conditions 1,3,5.
	 *  
	 */
	public int[] conditionSelection;

	/** True if all voxels (>= 0?) are to be included in masked data for corresponding
	 *  session; length of array == no. of sessions. 
	 *  If no mask name is provided and inclAllVoxels = false for a given sess file
	 *  then data is masked by using PLS-style thresholding technique.
	 *  
	 */
	// TODO: what happens to voxels <= 0 in NPAIRS analysis? 
	//	public Boolean[] inclAllVoxels;

	/** Array containing threshold values for each sess file
	 *  (to be used to determine data mask); if current sess
	 *  file has non-null mask filename or inclAllVoxels
	 *  is true, corresponding maskThreshVal is ignored;
	 *  otherwise it is used to calculate mask.
	 */
	//	public double[] maskThreshVals;

	/**
	 * Number of 3D image volumes (scans) in input data (excluding skipped
	 * scans)
	 */
	public int numVols;

	/** Integer array (length no. rows in input data) indicating run no. assoc. with each
	 *  data volume.
	 */
	private int[] runLabels;


	// **Data reduction options**

	/**
	 * True if data is to be transformed into new vector space via some sort of
	 * feature selection method before getting pumped through the NPAIRSJ
	 * analysis framework. Feature selection is commonly used to reduce
	 * dimensionality of the data before running analysis.
	 */
	public boolean initFeatSelect = false;

	/**
	 * EVD is one type of feature selection. (Can add boolean initWavelet etc
	 * later as required)
	 */
	public boolean initEVD = false;
	
	public boolean normedEVD = false;

	public boolean loadEVD = false;

	public String evdFilePref = "";

	/**
	 * Proportion of data to keep from initial feature selection (0.0 <
	 * dataReductionFactor <= 1.0) [Deprecated but kept for backwards compatibility]
	 */
	public double dataReductionFactor = 1.0;
	
	/** Number of PCs to keep from initial feature selection (1 - no. 
	 *  input tmpts)
	 */
	
	/** 

	// **Resampling options**

	/**
	 * True if data is to be resampled. Resampling involves partitioning the
	 * data repeatedly into disjoint training and test sets, running an analysis
	 * on the training set to generate a model, and then testing the model on
	 * the test data set. If false, runFullDataAnalysis must be set to true.
	 */	
	public boolean resampleData = false;

	/**
	 * If elements in split training and test sets are explicitly supplied by
	 * user, then input split info is contained here. Dims of 'splits' are
	 * [2][numSamples][]; splits[0] contains info for first split half and
	 * splits[1] contains info for second split half, and numSamples =
	 * min(numSplits, max poss splits). Array elements are indices into
	 * setupParams.splitObjectLabels array (in ascending order).
	 */
	public int[][][] splits = null;	

	/**
	 * True if resampling strategy includes switching roles of training and
	 * test subsets of data. Split-half cross-validation, e.g., requires
	 * that training and test sets be switched; bootstrap resampling, on the
	 * other hand, does not.
	 */
	public boolean switchTrainAndTestSets = false;

	/**
	 * Upper bound on number of times the data is to be partitioned into
	 * disjoint training and test sets in resampling framework. numSplits >= 0.
	 * If numSplits >= max possible number of disjoint splits, then total number
	 * of splits == max possible disjoint splits. If switchTrainAndTestSets ==
	 * true, then each 'split' of the data is used twice, hence data analysis is
	 * run numSplits*2 times on resampled data. Otherwise each split is used
	 * just once, hence data analysis is run numSplits times on resampled data.
	 */
	public int numSplits = 50;

	/**
	 * String containing name of file containing data stored in 'splits' array,
	 * i.e. volume numbers of images contained in each split half. If file
	 * exists, 'splits' info is loaded from it; otherwise FileNotFoundException
	 * is thrown.
	 */
	public String splitsInfoFilename = "";

	/**
	 * 2-element integer array containing number of split objects (e.g. subject
	 * or run) to include in each split of the data. E.g., {3,4} means each
	 * split will consist of 3 split objects in the first half, and 4 in the
	 * second half. Each split half must contain at least 1 split object, and
	 * less than total number of split objects. Split halves must be disjoint,
	 * so A + B <= total number of split objects, where numSplitObjInSplits =
	 * {A,B}.
	 */
	public int[] numSplitObjInSplits = new int[2];

	/**
	 * Integer array (length no. rows in input data) indicating to which group 
	 * each input scan belongs; to be used when partitioning data into splits 
	 * for resampling.
	 * Each split half will contain split objects from each group, in the same
	 * proportion as in the full dataset. (E.g., if 2/3 of the full dataset
	 * belongs to Group 1 and 1/3 to Group 2, then each split half will contain
	 * 2/3 Group 1 scans and 1/3 Group 2 scans.)
	 */
	private int[] groupLabels;

	/**
	 * Integer array (length no. rows in input data) indicating units to use when splitting
	 * data. E.g., if split object is subject, then all scans belonging to a
	 * given subject are given the same label in splitObjectLabels, and are
	 * treated as a single unit when partitioning the scans into 2 split halves.
	 */
	private int[] splitObjectLabels;

	/**
	 * Integer array (length no. rows in input data) indicating subject no. assoc. with each
	 * data volume. (required when saving IDL listfile - see saveListfile() - and
	 * also when saving into pls-style .mat file)
	 */
	private int[] subjectLabels;

	/**
	 * Integer array (length no. rows in input data) indicating session no. assoc. with each
	 * data volume. 
	 */
	private int[] sessionLabels;

	/**
	 * True if analysis is to be run on all the data samples at once, not just
	 * on resampled subsets. Must be set to true if no resampling is to be done
	 * (resampleData == false). Also set to true in many cases where resampling
	 * is performed, in order to provide some insights into variance structure
	 * of full data, and to provide reference for orientation of results when
	 * resampling.
	 */ 
	public boolean runFullDataAnalysis = true;

	/**
	 * True if full data analysis results are to be saved.  
	 * @see saveLotsOfFiles
	 */
	public boolean saveFullDataAnalysis = true;

	/**
	 * True if split data analysis results are to be saved.  
	 * @see saveLotsOfFiles
	 */
	public boolean saveSplitDataResults = true;


	// ** Data preprocessing options**

	/**
	 * True if data is to be preprocessed (i.e., intensity values normalized in
	 * some way, e.g. proportional scaling) before analysis. Note that
	 * preprocessing is performed before every data analysis, whether it's on
	 * full data or on a resampled training set, unlike initial feature
	 * selection/data reduction, which is performed just once on original full
	 * data set.
	 */
	public boolean preProcess = false;
	// + preproc vars:

	/** True if mean session effects (i.e., mean session scans) are to be removed
	 *  before analysis is run
	 */
	public boolean removeSessionMeans = false;

	// **GLM options (not currently implemented)**

	/**
	 * True if GLM is to be run on data. Not currently implemented.
	 */
	public boolean glmRun = false;
	// + glm vars

	// **PCA options**

	/**
	 * True if PCA is to be run on data. Can be run on its own as analysis
	 * technique, or used as data reduction technique before doing CVA analysis.
	 * If a CVA is run, then it must be preceded by a PCA.
	 */
	public boolean pcaRun = true;

	/** True if PC Eigenimages are to be transformed back into original space before saving.
	 *  Default is pcEigimsToBigSpace = false.
	 *  This is a very time-consuming step.
	 */
	boolean pcEigimsToBigSpace = false;

	/** True if denoised data (i.e. data entered into CVA after PCA) is to be 
	 *  saved (in original image space) as 4D nifti file.
	 *  Default is false. 
	 */
	boolean saveDataPostPCA = false;

	/** True if PC Scores are to be normalized by dividing each PC dimension by
	 *  its standard deviation (i.e. so each dim of PC Scores has variance 1).
	 *  Default is not to normalize.
	 */
	public boolean normalizePCsBySD = false;


	/**
	 * True if results from PCA are to be kept on disk. This will be the case by
	 * default if PCA is the final analysis step. If PCA is a precursor to CVA,
	 * however, then interim PCA results will not be saved by default.
	 *
	 */
	public boolean savePCAResults = false;

	// **CVA options**

	/**
	 * True if CVA is to be run on data. Preceded by a PCA to reduce the
	 * dimensionality of the data before running the CVA on it.
	 */
	public boolean cvaRun = true;

	/**
	 * True if results from CVA are to be kept on disk. This will be the case by
	 * default.
	 */
	public boolean saveCVAResults = true;

	/**
	 * Array (length no. of rows in input data) containing class
	 * label for corresponding row. Each class group must contain at
	 * least 2 samples (scans/rows). E.g. if input data consists of 12 scans
	 * containing 3 off/on epochs of equal length, and we wish to define 2
	 * classes, an 'off' class and an 'on' class, then the corresp. cvaClassDef
	 * array will look like this: {0,0,1,1,0,0,1,1,0,0,1,1}. (Don't need to use
	 * 0s and 1s; can label the 2 conditions using any 2 integers.)
	 */
	private int[] classLabels;

	/** Number of NPAIRS analyses to run [can only be > 1 if PC Range is chosen
	 *  in Analysis setup; multiple analyses are to be run varying no. of
	 *  PCs to be passed into CVA].
	 * 
	 */
	public int numNPAIRS = 1;
	
	/** If true, set PC range to enter into CVA instead of using explicitly
	 *  given set of PC components for split and full data analyses
	 */
	public boolean setPCrange = false;

	/** If setPCrange, pcMultFactor is proportion of split-data
	 *  PC set size to use in full-data CVA (e.g. if pcMultFactor is 2.0, use
	 *  twice as many PCs in full-data CVA as in split-data CVA).
	 */
	public double pcMultFactor = 1.0;

	/** If setPCrange, pcStep gives the step size between each
	 *  successive number of PCs to use in the analyses (e.g. if 
	 *  pcRangeSplit = [10,23] and pcStep = 5 then 3 analyses will be run;
	 *  PC split-data sets will be 1-10, 1-15 and 1-20.)
	 */
	public int pcStep = 1;

	/** Number of PCs in first split PC set (e.g. if pc range = [10-20],
	 *  minPCsSplit = 10). Set to no. of PC components in split analysis if 
	 *  no PC range set.
	 */
	public int minPCsSplit;

	/** Number of PCs in first full-data PC set (e.g. if pc range = [10-20]
	 *  and pcMultFactor = 1.3, minPCsFull = 13). Set to no. of PC components 
	 *  in full-data analysis if no PC range set.
	 */
	public int minPCsFull;

	/**
	 * Set of indices (0-relative) indicating which components (dimensions) are
	 * to be passed on to CVA from PCA of first (original training) split half.
	 * E.g., if first 10 components are to be passed into CVA,
	 * cvaPCASet1 = {0,1,2,3,4,5,6,7,8,9}. Note that this variable must be set
	 * whenever CVA is performed in a resampling framework; but cvaPCASet2 need
	 * only be set if the resampling framework includes switchTrainAndTestSets ==
	 * true.
	 * 
	 * @see cvaPCSet2
	 * @see cvaPCSetAll
	 */
	public int[] cvaPCSet1 = {0};

	/**
	 * Set of indices (0-relative) indicating which components (dimensions) are
	 * to be passed on to CVA from PCA of second (originally test, now training)
	 * resampled data subset. Note that this is only relevant if the resampling
	 * framework being used includes switchTrainAndTestSets == true.
	 * 
	 * @see cvaPCSet1
	 * @see cvaPCSetAll
	 */
	public int[] cvaPCSet2 = {0};

	/**
	 * Set of indices (0-relative) indicating which components (dimensions) are
	 * to be passed on to CVA from PCA of full data. Note that this must be set
	 * whenever a full data CVA is performed.
	 * 
	 * @see cvaPCSet1
	 * @see cvaPCSet2
	 */
	public int[] cvaPCSetAll = null;

	/**
	 * Prefix to use for all results files.
	 */

	public String resultsFilePrefix;

	/** Same as resultsFilePrefix unless setPCrange is true; 
	 *  then resultsFilePrefix = baseResultsFilePrefix + specific info string
	 */
	public String baseResultsFilePrefix;

	/** Save many results files (text and .img/hdr files (for prediction 3d vols)).
	 *  If false, only save .mat result file, .vols split info file,
	 *  init. EVD files and prediction/reprod. files for plotting R vs P curves.
	 *  Default is to save lots of files (full-data and summary results; split
	 *  data results are only saved if 'saveSplitDataAnalysis' and 'saveLotsOfFiles'
	 *   == true). 
	 *  @see saveSplitDataAnalysis 
	 *  @see saveFullDataAnalysis
	 */
	protected boolean saveLotsOfFiles = true;

	/** True if session file Conditions are to be used as CVA classes (default);
	 *  false if CVA classes are to be loaded via text file instead.
	 */
	public boolean useCondsAsClasses = true;

	transient private MLStructure npairsSetupParamStruct = null;

	/** True if NPAIRS analysis is 'event-related' (i.e., if data is to be 
	 *  loaded via datamats instead of image files).  Note that an 'event-
	 *  related' NPAIRS can actually be run on block data as well,
	 *  since datamats can be created from block data.
	 */
	private boolean loadDatamats;

	/** True if sess files are block (i.e. end in "_BfMRIsession.mat")	 * 
	 */
	public boolean blockSessFiles;
	
	private boolean calcProcWithEigims = false; // if false, use mean CV scores instead to
	                                            // CV eigenimages to calculate Procrustes transform
	                                            // when aligning split results to full-data ref. results
	public static final int AUTO_PC_STEP_SIZE = 1; // pc step size when setting PC range automatically
	
	/**************************************************************************** */


	/**
	 * Constructor - reads in parameters from input .mat file and sets all
	 * variables required to run NPAIRSJ.
	 * 
	 * @param paramMatFilename
	 * @param eventRelAnalysis - if event-related analysis, data is loaded from datamats
	 * 								instead of from input image files; scans data is averaged
	 * 								for each condition (class) within session/subject.
	 */
	public NpairsjSetupParams(String paramMatFilename, boolean eventRelAnalysis) throws NpairsjException, IOException {

		//		Npairsj.output.println("Reading NPAIRS analysis setup file " + paramMatFilename + "...");

		try {
			npairsSetupParamStruct = (MLStructure)new NewMatFileReader(paramMatFilename).
				getContent().get("npairs_setup_info");
			
//			npairsSetupParamStruct.setField("num_splits", new MLDouble("num_splits", new double[][]{{1}}));

			
		} 
		catch (Exception e) {
			throw new IOException("NPAIRS analysis setup file " + 
					paramMatFilename + " could not be loaded.");
		}

		this.loadDatamats = eventRelAnalysis;

		setSaveResultsInfo();

		setSessionFileInfo();

		setResamplingInfo();

		setInitFeatSelInfo();

		setAnalysisModelInfo();

	}

	/** Initializes log file.
	 * 
	 * @param evdOnly if true, log file has suffix .EVD.log
	 * @param analysisNum 1-relative. If > 0 then get no. of pcs N to be used
	 *                    in split data analysis and append '_Npc' to log file
	 *                    prefix, so each analysis has its own log file.
	 *                    If analysisNum == 1 and no pc range has been set then
	 *                 	  log file is not given pc extension because this analysis
	 *                    is assumed to be the only one.
	 */
	public void initLogFile(boolean evdOnly, int analysisNum) {
		String suffix = "";
		if (evdOnly) {
			suffix = ".EVD.log";
		}
		else {
			suffix = ".analysis.log";
		}
		String logFilePrefix = resultsFilePrefix;
		if (analysisNum > 0) {
			int currMaxPCSplit = minPCsSplit + ((analysisNum - 1) * pcStep);
			String formattedPCString = "";
			if (analysisNum == 1 && currMaxPCSplit > 0) {
				formattedPCString = String.format("_%03dpc", currMaxPCSplit);
			}
			logFilePrefix = logFilePrefix.concat(formattedPCString);
		}
		try {
			Npairsj.output = new PrintStream(logFilePrefix + suffix);
		}
		catch (Exception e) {
			// Print to standard out if can't create a log file.
			Npairsj.output = System.out;
			Npairsj.output.println("Couldn't create log file - writing trace to System.out");
		}
	}


	private void setSessionFileInfo() throws IOException, FileNotFoundException,
	NpairsjException {

		NpairsSetupSessInfo setupSessInfo = new NpairsSetupSessInfo(npairsSetupParamStruct, loadDatamats);
		blockSessFiles = setupSessInfo.blockSessFiles;
		sessInfo = setupSessInfo.getSessionInfo();
		nDataFiles = setupSessInfo.getNDataFiles();
		nSkipTmpts = setupSessInfo.getNSkipTmpts();
		numVols = setupSessInfo.getNumVols();
		groupLabels = setupSessInfo.getGroupLabels();
		conditionSelection = setupSessInfo.getCondSelection();
		sessionLabels = setupSessInfo.getSessLabels();
		subjectLabels = setupSessInfo.getSubjLabels();
		runLabels = setupSessInfo.getRunLabels();		
		classLabels = setupSessInfo.getClassLabels();
		useCondsAsClasses = setupSessInfo.useCondsAsClasses();
		
	}


	private void setSaveResultsInfo() {
		// get results save info
		double d = ((MLDouble)npairsSetupParamStruct.getField("save_multi_files")).
		get(0,0).doubleValue();
		saveLotsOfFiles = (d == 1);
		try {
			double d2 = ((MLDouble)npairsSetupParamStruct.getField("save_split_results")).
			get(0,0).doubleValue();
			saveSplitDataResults = (d2 == 1);
		}
		catch (NullPointerException ne) {
			// save split results not implemented yet
		};

		try {
			double d3 = ((MLDouble)npairsSetupParamStruct.getField("pc_eigims_in_img_space")).
			get(0,0).doubleValue();
			pcEigimsToBigSpace = (d3 == 1);
		}
		catch (NullPointerException ne) {
			// pc eigims to big space menu option not implemented yet 
		};

		try {
			double d4 = ((MLDouble)npairsSetupParamStruct.getField("save_data_post_pca")).
			get(0,0).doubleValue();
			saveDataPostPCA = (d4 == 1);
		}
		catch (NullPointerException ne) {
			// save denoised (post-PCA) data option not implemented yet
		};

		String resultsFileName = ((MLChar)npairsSetupParamStruct.getField(
		"results_filename")).getString(0);
		resultsFilePrefix = resultsFileName;
		if (resultsFileName.endsWith(NpairsfMRIResultFileFilter.EXTENSION)) {
			resultsFilePrefix = resultsFileName.substring(0, resultsFileName.
					indexOf(NpairsfMRIResultFileFilter.EXTENSION));
		}
		baseResultsFilePrefix = resultsFilePrefix;
	}


	private void setResamplingInfo() throws NpairsjException, IOException, FileNotFoundException {
		int doSplitHalfResampling = ((MLDouble)npairsSetupParamStruct.getField(
		"split_half_xvalid")).get(0,0).intValue();	
		int doBootstrap = ((MLDouble)npairsSetupParamStruct.getField("bootstrap")).
		get(0,0).intValue();
		if (doSplitHalfResampling == 1) {
			resampleData = true;
			switchTrainAndTestSets = true;
		}
		if (doBootstrap == 1) {
			resampleData = true;
			switchTrainAndTestSets = false;
		}
		numSplits = ((MLDouble)npairsSetupParamStruct.getField("num_splits")).
		get(0,0).intValue();
		getSplitInfoFilename();

		splitObjectLabels = getSplitObjLabels();
		int numSplitObj = MLFuncs.unique(splitObjectLabels).length;
		numSplitObjInSplits = getSplitPartInfo(numSplitObj);
	}


	public int[] getSplitObjLabels() {
		int[] splitObjLabels = null;
		try {
			String splitObjType = getSplitObjType();
			if (splitObjType.equals("Session (default)")) {
				splitObjLabels = getSessLabels();
			}
			else if (splitObjType.equals("Run")) {
				splitObjLabels = getRunLabels();
			}
		} 
		catch (NullPointerException npe) {
			// split object type wasn't implemented yet so go with 
			// default split object (session)
			splitObjLabels = getSessLabels();
		}

		return splitObjLabels;
	}


	private int[] getRunLabels() {
		return runLabels;
	}

	public int[] getClassLabels() {
		return classLabels;
	}

	/** @return split object type - either "Run" or "Session (default)"
	 * 
	 */
	public String getSplitObjType() {
		return ((MLChar)npairsSetupParamStruct.getField("split_type")).getString(0);
	}

	private int[] getSplitPartInfo(int numSplitObj) {
		int[] splitPartition = new int[2];
		try {
			MLDouble splitInfoDbl = ((MLDouble)npairsSetupParamStruct.getField("split_partition"));
			splitPartition[0] = splitInfoDbl.get(0,0).intValue();
			splitPartition[1] = splitInfoDbl.get(0,1).intValue();

		}
		catch (NullPointerException npe) {
			// split partition info not included in param file; use default values
			splitPartition[1] = numSplitObj/2;
			splitPartition[0] = numSplitObj - splitPartition[1];
		}
		return splitPartition;
	}


	private void setAnalysisModelInfo() throws NpairsjException {
		try {
			int doMSR = ((MLDouble)npairsSetupParamStruct.getField("do_msr")).
			get(0,0).intValue();
			if (doMSR == 1) {
				preProcess = true;
				if (debug) {
					Npairsj.output.println("Setting MSR = true...");
				}
				removeSessionMeans = true;
			}
		}
		catch (NullPointerException npe) {
			// MSR wasn't implemented yet when analysis setup file being read in was created
			if (debug) {
				Npairsj.output.println("no MSR info in file so MSR = false");
			}
		}
		int doGLM = ((MLDouble)npairsSetupParamStruct.getField("do_glm")).
		get(0,0).intValue();
		if (doGLM == 1) {
			glmRun = true;
		}
		int doPCA = ((MLDouble)npairsSetupParamStruct.getField("do_pca")).
		get(0,0).intValue();
		if (doPCA == 1) {
			pcaRun = true;
			// normalize PCs?
			normalizePCsBySD = (((MLDouble)npairsSetupParamStruct.getField(
			"norm_pcs")).get(0,0).intValue() == 1);	
		}

		int doCVA = ((MLDouble)npairsSetupParamStruct.getField("do_cva")).
		get(0,0).intValue();
		if (doCVA == 1) {
			cvaRun = true;
			if (resampleData) {
				// how to match CV results to ref?
				try {
					int useEigims = ((MLDouble)npairsSetupParamStruct.getField(
							"eigim_procrustes")).get(0,0).intValue();
					if (useEigims == 1) {
						calcProcWithEigims = true;
					}
				} catch (NullPointerException npe) { }// not implemented yet; OK 
						
			}
			if (pcaRun) {
				getPCSetsForCVA();
			}
		}
		else {
			if (pcaRun) {
				savePCAResults = true;
			}
		}
	}

	private void setInitFeatSelInfo() {
//		boolean initEVD = false;
		try {
			initEVD = (((MLDouble)npairsSetupParamStruct.getField("do_init_svd")).
					get(0,0).intValue() == 1);
			if (initEVD) {
				initFeatSelect = true;
				dataReductionFactor = ((MLDouble)npairsSetupParamStruct.getField(
				"drf")).get(0,0); 
				loadEVD = (((MLDouble)npairsSetupParamStruct.getField("load_svd")).
						get(0,0).intValue() == 1);
				if (loadEVD) {
					evdFilePref = ((MLChar)npairsSetupParamStruct.getField(
					"svd_file_prefix")).getString(0);
				}
			}
		}
		catch (NullPointerException npe) {
			// Field names in .mat file use 'evd' as of v. 1.1.6 instead of 'svd';
			// bkwds compatibility has been kept 
			initEVD = (((MLDouble)npairsSetupParamStruct.getField("do_init_evd")).
					get(0,0).intValue() == 1);
			if (initEVD) {
				normedEVD = (((MLDouble)npairsSetupParamStruct.getField("norm_init_evd")).
					get(0,0).intValue() == 1);
				initFeatSelect = true;
				dataReductionFactor = ((MLDouble)npairsSetupParamStruct.getField(
				"drf")).get(0,0); 
				loadEVD = (((MLDouble)npairsSetupParamStruct.getField("load_evd")).
						get(0,0).intValue() == 1);
				if (loadEVD) {
					evdFilePref = ((MLChar)npairsSetupParamStruct.getField(
					"evd_file_prefix")).getString(0);
				}
			}			
		}
	}


	private void getPCSetsForCVA() throws NpairsjException {
		try {
			// get pc sets for full and split-data analyses
			String pcsAllData = ((MLChar)npairsSetupParamStruct.getField(
			"pcs_all_data")).getString(0);
			cvaPCSetAll = getPCSet(pcsAllData);
			minPCsFull = cvaPCSetAll[cvaPCSetAll.length - 1] + 1; // for compatibility with pc range set option

			if (resampleData) {
				String pcsTraining = ((MLChar)npairsSetupParamStruct.getField(
				"pcs_training")).getString(0);
				cvaPCSet1 =  getPCSet(pcsTraining);
				cvaPCSet2 = cvaPCSet1;
				minPCsSplit = cvaPCSet1[cvaPCSet1.length - 1] + 1; // set to max PC component for compatibility with pc range set option
				
			}
		}
		catch (NullPointerException e) {
			// pc range must be set instead
			setPCrange = true;
			int[] pcRangeSplit = new int[2];
			boolean setAutoPCRange=false;
			/*boolean setAutoPCRange = ((MLDouble)npairsSetupParamStruct.getField(
					"set_auto_pc_range")).get(0,0).intValue() == 1;*/
			if (setAutoPCRange == true) {
				// set PC range automatically to [2 - max no. possible PC dims] 
				// for both full and split data 
				pcRangeSplit[0] = 2;
				pcRangeSplit[1] = getMaxPCsFullData();				
			}
			else {
				// set pc range for split data
				String splitPCRange = ((MLChar)npairsSetupParamStruct.getField(
				"pc_range")).getString(0);
				String pcRangeFormat = "\\s*\\d+\\s*-\\s*\\d+\\s*";
				if (splitPCRange.matches(pcRangeFormat)) {
					int dashLoc = splitPCRange.indexOf("-");
					String pcFirst = splitPCRange.substring(0, dashLoc);
					String pcLast = splitPCRange.substring(dashLoc + 1);
					Integer firstPC = Integer.decode(pcFirst.trim());
					Integer lastPC = Integer.decode(pcLast.trim());
					checkRangeFormat(splitPCRange, firstPC, lastPC);

					pcRangeSplit[0] = firstPC;
					pcRangeSplit[1] = lastPC;						
				}
				else {
					throw new NpairsjException("Incorrect PC range syntax");
				}

				pcMultFactor = ((MLDouble)npairsSetupParamStruct.getField(
				"pc_mult_factor")).get(0,0).doubleValue();
			}
			pcStep = ((MLDouble)npairsSetupParamStruct.getField(
				"pc_step")).get(0,0).intValue();
			if (pcStep > 0) {
			numNPAIRS = ((pcRangeSplit[1] - pcRangeSplit[0]) / 
					pcStep) + 1;
			
				System.out.println("Alan: PCStep = " + pcStep);
				System.out.println("Alan: pcRangeSplit[0] = " + pcRangeSplit[0]);
				System.out.println("Alan: pcRangeSplit[1] = " + pcRangeSplit[1]);
			}
			else throw new NpairsjException("PC range step must be > 0! Aborting analysis.");
			
			minPCsSplit = pcRangeSplit[0];		
			minPCsFull = (int)Math.round(minPCsSplit * pcMultFactor);
		}
	}


	private void getSplitInfoFilename() throws NpairsjException, IOException, FileNotFoundException {
		try {
			splitsInfoFilename = ((MLChar)npairsSetupParamStruct.getField(
			"splits_info_filename")).getString(0);
		}
		catch (NullPointerException e) {
			// no splits info filename entered
		}

		if (!splitsInfoFilename.equals("")) {
			if (!splitsInfoFilename.contains(".")) {
				splitsInfoFilename = splitsInfoFilename.concat(".vols");
			}
			File splitsInfoFile = new File(splitsInfoFilename);
			if (splitsInfoFile.exists()) {
				//		Npairsj.output.println("Loading splits info from file " + splitsInfoFilename);
				loadSplitsInfo();
			}
			else {
				throw new FileNotFoundException("Splits info file " + splitsInfoFilename 
						+ " does not exist");
			}
		}
	}


	/**
	 * Parses input formatted String containing PC set (1-relative) and returns
	 * 0-relative array of pc indices
	 * 
	 * @param inputPCSet -
	 *            formatted String containing PC set - String must be of form
	 *            "1,3-7,10-15,20", with or without whitespace
	 * @return int array containing indices corresponding to input set of PCs
	 * @throws NpairsjException
	 *             if input String is incorrectly formatted
	 */
	private int[] getPCSet(String inputPCSet) throws NpairsjException {

		String[] pcElems = inputPCSet.split(",");
		String pcRangeFormat = "\\s*\\d+\\s*-\\s*\\d+\\s*";
		Vector<Integer> pcSet = new Vector<Integer>();

		for (String nextElement : pcElems) {

			if(debug) {
				Npairsj.output.println("[" + nextElement + "]");
			}

			if (nextElement.matches(pcRangeFormat)) {
				int dashLoc = nextElement.indexOf("-");
				String pcFirst = nextElement.substring(0, dashLoc);
				String pcLast = nextElement.substring(dashLoc + 1);
				Integer firstPC = Integer.decode(pcFirst.trim());
				Integer lastPC = Integer.decode(pcLast.trim());

				checkRangeFormat(nextElement, firstPC, lastPC);

				for (int i = 0; i < lastPC - firstPC + 1; ++i) {
					if (!pcSet.contains(firstPC + i)) {
						pcSet.add(firstPC + i);
					}
					else {
						throw new NpairsjException("Incorrect PC set syntax - "
								+ "\ncannot include the same PC more than once");
					}
				}
			}
			else {	
				try {
					Integer pcAsInt = Integer.decode(nextElement.trim());
					if (pcAsInt <= 0) {
						throw new NpairsjException("Incorrect PC set syntax - "
								+ "PCs must be >= 0");
					}
					if (!pcSet.contains(pcAsInt)) {
						pcSet.add(pcAsInt);
					}
					else {
						throw new NpairsjException("Incorrect PC set syntax - "
								+ "\ncannot include the same PC more than once");
					}
				}
				catch (NumberFormatException e) {
					throw new NpairsjException(e.getMessage() +
					"\nIncorrect PC set syntax");
				}
			}
		}

		int[] pcIndices = new int[pcSet.size()];
		for (int i = 0; i < pcIndices.length; ++i) {
			pcIndices[i] = pcSet.get(i) - 1;		
		}
		pcIndices = MLFuncs.sortAscending(pcIndices);

		if (debug) {
			Npairsj.output.println("PC Index Array: ");
			NpairsjIO.print(pcIndices);	
		}

		return pcIndices;
	}


	/** Checks that input expression pcRangeExp is of format A-B (with or without whitespace)
	 * 
	 * @param pcRangeExp
	 * @param firstPC
	 * @param lastPC
	 * @throws NpairsjException if format is incorrect
	 */
	private void checkRangeFormat(String pcRangeExp, Integer firstPC, Integer lastPC) 
	throws NpairsjException {
		String rangeFormatErrorMessage = "Incorrect PC set syntax (\"" +
		pcRangeExp + "\") - \nrange format must be \"A-B\", " + 
		"where A < B and A, B > 0";
		if (firstPC <= 0) {
			throw new NpairsjException(rangeFormatErrorMessage);
		}
		if (lastPC < firstPC) {
			throw new NpairsjException(rangeFormatErrorMessage);
		}
	}


	/** Sets 'splits' variable
	 * 
	 * @throws NpairsjException
	 * @throws IOException
	 */
	private void loadSplitsInfo() throws NpairsjException, IOException {
		// splits2D is 1-relative
		int[][] splits2D = NpairsjIO.readIntsFromIDLFile(splitsInfoFilename);
		int numSamples = splits2D.length / 2;
		int nSplitVols = splits2D[0].length;

		// splits is 0-relative but IDL splits info file is 1-relative
		splits = new int[2][numSamples][];
		Vector<Integer> currSplitVols = new Vector<Integer>(); 
		for (int s = 0; s < (numSamples * 2); ++s) {
			currSplitVols.clear();
			for (int v = 0; v < nSplitVols; ++v) {
				if(splits2D[s][v] == 0) {
					break;
				}
				currSplitVols.add(splits2D[s][v] - 1);
			}
			int[] intCurrVols = new int[currSplitVols.size()];
			for (int i = 0; i < intCurrVols.length; ++i) {
				intCurrVols[i] = currSplitVols.get(i);
			}
			int currSamp = s / 2;
			splits[(s % 2)][currSamp] = intCurrVols;
		}
	}


	//	/**
	//	 * Saves record of input params in 'listfile' text format used in IDL
	//	 * NPAIRS.
	//	 * Saves in file "resultsFilePrefix.CVA.ALL.info".
	//	 * 
	//	 */
	//	public void saveListfile() throws IOException {
	//		if (!dataIs4D) {
	//			String creationInfo = "Created on " + getDateTime() + " by '" + 
	//			System.getProperty("user.name") + "'";
	//			String fieldNames = "N \t\t Volume \t\t Mask \t\t Popul  Protcl  Subject  Session  " +
	//			"Scan  Trial  State  Age  Sex  Weight  Dose  Misc";
	//			String decorativeLine = new String();
	//			for (int i = 0; i < fieldNames.length() + 16; ++i) {
	//				decorativeLine = decorativeLine.concat("=");
	//			}
	//
	//			// TODO: For now, save as 'blah.CVA.ALL.info'. Will need to change for
	//			// resampled data and maybe add 'blah.PCA.ALL.info' for viewing
	//			// pc results using idl tools
	//			String fileName = resultsFilePrefix + ".CVA.ALL.info";
	//			PrintWriter pw = null;
	//			try {
	//				pw = new PrintWriter(new BufferedWriter(new FileWriter(fileName)));
	//
	//				pw.println("#");
	//				pw.println("# " + creationInfo);
	//				pw.println("#");
	//				pw.println("#==" + decorativeLine);
	//				pw.println("#  " + fieldNames);
	//				pw.println("#==" + decorativeLine);
	//
	//
	//				// add each row of data
	//				for (int volNum = 0; volNum < numVols; volNum++) {
	//					String nextVolNum = Integer.toString(volNum + 1);
	//					String nextVolFilename = stripFilePath(getDataFilenames()[volNum]);
	//					String nextMaskFilename = stripFilePath(maskFileNames[volNum]);
	//					String nextPopul = "0";
	//					String nextProtcl = "0";
	//					String nextSubject = Integer.toString(subjectLabels[volNum]);
	//					String nextSession = Integer.toString(sessionLabels[volNum]);
	//					String nextScan = Integer.toString(volLabels[volNum]);
	//					String nextTrial = "0";
	//					String nextState = "0";
	//					String nextAge = "0";
	//					String nextSex = "U";
	//					String nextWeight = "0.0";
	//					String nextDose = "0.0";
	//					String nextMisc = Integer.toString(cvaClassLabels[volNum]);
	//
	//					String nextRow = "\t" + nextVolNum + "\t\t" + nextVolFilename + 
	//					"\t\t" + nextMaskFilename + "\t" + nextPopul + "\t" + nextProtcl
	//					+ "\t" + nextSubject + "\t" + nextSession + "\t" + nextScan + 
	//					"\t" + nextTrial + "\t" + nextState + "\t" + nextAge + "\t" + 
	//					nextSex + "\t" + nextWeight + "\t" + nextDose + "\t" + nextMisc;
	//
	//					pw.println(nextRow);
	//				}
	//			} 
	//			catch (IOException e) {
	//				e.printStackTrace();
	//			}
	//			finally {
	//				try {
	//					if (pw != null) {
	//						boolean error = pw.checkError();
	//						if (error) {
	//							throw new IOException("Error occurred writing to file "
	//									+ fileName);
	//						}
	//						pw.close();
	//					}
	//				}
	//				catch (IOException e) {
	//					throw new IOException(e.getMessage());
	//				}
	//
	//				catch (Exception e) {
	//					throw new IOException("Error - could not close file " + fileName);
	//				}
	//			}	
	//		}
	//		else {
	//			// input data is in 4D format; incompatible with IDL NPAIRS so don't
	//			// bother saving 'listfile'
	//		}
	//	}

	/** @return max possible number of components (dimensions) in input data after
	 *  initial EVD and data reduction (rounded to nearest no. components)
	 */
	public int getMaxPCsFullData() {
		return (int)Math.round(numVols * dataReductionFactor);
	}

	/** sets CVA PC sets for given NPAIRS analysis
	 * 
	 * @param analysisNum - current NPAIRS analysis number (0-relative)
	 *                      (0 <= analysisNum < numNPAIRS)
	 * @param helper stream to which progress messages are to be printed (e.g.
	 * 					GUI progress dialog or command-line console)
	 * @throws NpairsjException if no PCs set because none in range
	 * @return true if all given PCs in range; false if some out of range
	 */
	public boolean setPCs(int analysisNum, StreamedProgressHelper helper) throws NpairsjException {
		
		int currMaxPCSplit = minPCsSplit + (analysisNum * pcStep);
		int currMaxPCFull =  minPCsFull + (analysisNum * 
				(int)Math.round(pcMultFactor * pcStep));
			
		// check PC numbers not out of range
		boolean pcsOutOfRange = false;
		int maxPossPCsSplit = getMaxPCsSplitData();
		System.out.println("\nMax poss PCs split data: " + maxPossPCsSplit);
		int maxPossPCsFull = getMaxPCsFullData();
		System.out.println("Max poss PCs full data: " + maxPossPCsFull);
		if (initFeatSelect) {
			if (currMaxPCSplit > maxPossPCsSplit) {
				String warnMsgSplit = "WARNING - out-of-range PC components for CVA (split data)." +
					"\nKeeping only PCs <= " + maxPossPCsSplit + " (max possible no.).";
				Npairsj.output.println(warnMsgSplit);
				helper.postMessage("\n" + warnMsgSplit + "\n");
				currMaxPCSplit = maxPossPCsSplit;
				pcsOutOfRange = true;
			}
			if (currMaxPCFull > maxPossPCsFull) {
				String warnMsgFull = "WARNING - out-of-range PC components for CVA (full data)." + 
					"\nKeeping only PCs <= " + maxPossPCsFull + " (max possible no.).";
				Npairsj.output.println(warnMsgFull);
				helper.postMessage("\n" + warnMsgFull + "\n");
				currMaxPCFull = maxPossPCsFull;
				pcsOutOfRange = true;
			}			
		}
		else {
			// TODO: check PC boundaries if init feat selection not performed (first need to
			// know how many (masked) voxels in input data).
		}

		if (setPCrange) { // truncate PC components and set 
			int[] currPCsSplit = new int[currMaxPCSplit];
			for (int k = 0; k < currMaxPCSplit; ++k) {
				currPCsSplit[k] = k;
			}
			cvaPCSet1 = currPCsSplit;
			cvaPCSet2 = currPCsSplit;

			int[] currPCsFull = new int[currMaxPCFull];
			for (int k = 0; k < currMaxPCFull; ++k) {
				currPCsFull[k] = k;
			}
			cvaPCSetAll = currPCsFull;
		}
		else { // eliminate given PC components that are out of range
			Vector<Integer> vCVAPCSetSplit = new Vector<Integer>();
			boolean splitRangeValid = true;
			for (int pc : cvaPCSet1) {
				if (pc < maxPossPCsSplit) {
					vCVAPCSetSplit.add(pc);
				}
				else {
					splitRangeValid = false;		
				}
			}
			if (!splitRangeValid) {
				int numInRange = vCVAPCSetSplit.size();
				cvaPCSet1 = new int[numInRange];
				for (int i = 0; i < numInRange; ++i) {
					cvaPCSet1[i] = vCVAPCSetSplit.get(i);
				}
				cvaPCSet2 = cvaPCSet1;
			}
			Vector<Integer> vCVAPCSetAll= new Vector<Integer>();
			boolean fullRangeValid = true;
			for (int pc : cvaPCSetAll) {
				if (pc < maxPossPCsFull) {
					vCVAPCSetAll.add(pc);
				}
				else {
					fullRangeValid = false;
				}
			}
			if (!fullRangeValid) {
				int numInRange = vCVAPCSetAll.size();
				cvaPCSetAll = new int[numInRange];
				for (int i = 0; i < numInRange; ++i) {
					cvaPCSetAll[i] = vCVAPCSetAll.get(i);
				}
			}
		}

		if (cvaPCSetAll.length == 0 || cvaPCSet1.length == 0 || cvaPCSet2.length == 0) {
			throw new NpairsjException("Error - no PCs in range.");
		}
		
		if (debug) {
			System.out.println("PC set split:");
			NpairsjIO.print(cvaPCSet1);
			System.out.println("PC set full: ");
			NpairsjIO.print(cvaPCSetAll);
			System.out.println("Results file prefix: ");
			System.out.println(resultsFilePrefix);
		}
		
		
		return pcsOutOfRange;
	}

	/** @return min possible no. of vols in each split half for given split object 
 	 * type and min no. split objects in partition
 	 */
	private int getMaxPCsSplitData() {
		// Find min no. vols in each split object
		int numSessFiles = sessInfo.length;
		Vector<Integer> minNumSplObjVols = new Vector<Integer>();
		for (int i = 0; i < numSessFiles; ++i) {
			if (getSplitObjType().equals("Run")) {
				minNumSplObjVols.addAll(sessInfo[i].getNumRunVols());
			}
			else if (getSplitObjType().equals("Session (default)")) {
				minNumSplObjVols.add(sessInfo[i].getNumVols());
			}
		}
		// Find smallest possible no. vols in given partition of split objects
		// 
		Integer[] iMinNumSplObjVols = new Integer[minNumSplObjVols.size()];
		int[] sortedNumVols = MLFuncs.sortAscending((Integer[])minNumSplObjVols.
				toArray(iMinNumSplObjVols));
		int minPartitionSz = Math.min(numSplitObjInSplits[0], numSplitObjInSplits[1]);
		// Add number of vols in smallest 'minPartitionSz' split objects
		int minVolsInSplitHalf = MLFuncs.sum(MLFuncs.getItemsAtIndices(sortedNumVols,
				MLFuncs.range(0, minPartitionSz - 1)));
		// max PCs in split half can't be greater than no. of vols in full data
		// after data reduction factor:
		int maxPCs = Math.min(minVolsInSplitHalf, getMaxPCsFullData());
		return maxPCs;
	}

	/** Sets results file prefix for given NPAIRS analysis
	 *  (given pc range)
	 * @param analysisNum - current NPAIRS analysis number (0-relative)
	 *                      (0 <= analysisNum < numNPAIRS)
	 *        
	 */
	public void setResultsFilePref(int analysisNum) {
		int currMaxPCSplit = minPCsSplit + (analysisNum * pcStep);
		if (getMaxPCsSplitData() < currMaxPCSplit) {
			currMaxPCSplit = getMaxPCsSplitData();
		}
		
		String pattern = "000";	
		DecimalFormat formatter = (DecimalFormat)DecimalFormat.getInstance();
		formatter.applyPattern(pattern);
		String zeroPadMaxPCSplit = formatter.format(currMaxPCSplit);
		resultsFilePrefix = baseResultsFilePrefix + "_" + zeroPadMaxPCSplit + "pc";
	}

	/** For testing */
	public static void main(String[] args) {
		String nspFile = "/home/anita/plsnpairs/grady/setup/" +
		"o2_fake3run_3d4dmixed_NPAIRSAnalysisSetup.mat";
		try {
			NpairsjSetupParams nsp = new NpairsjSetupParams(nspFile, false);
			nsp.printParams();
		}
		catch (Exception e) {
			e.printStackTrace();
		}

	}

	private void printParams() {
		//		Npairsj.output.println("Dataformat: " + dataFormat);
		Npairsj.output.println("No. data files: " + nDataFiles);
		Npairsj.output.println("Datafilenames: ");
		for (String dfn : getDataFilenames()) { 
			Npairsj.output.print(dfn + " ");
		}
		Npairsj.output.println();
		Npairsj.output.println("No. mask files: " + getMaskFilenames().length);
		Npairsj.output.println("Maskfilenames: ");
		for (String mfn : getMaskFilenames()) { 
			Npairsj.output.print(mfn + " ");
		}
		Npairsj.output.println();
		Npairsj.output.println("Include all voxels? ");
		NpairsjIO.print(inclAllVoxels());
		Npairsj.output.println();
		Npairsj.output.println("Mask thresh vals: ");
		NpairsjIO.print(getMaskThreshVals());
		Npairsj.output.println("Num vols: " + numVols);
		Npairsj.output.println();

		Npairsj.output.println("Length sessionLabels: " + getSessLabels().length);
		Npairsj.output.println("Sess labels: ");
		NpairsjIO.print(getSessLabels());
		Npairsj.output.println("Length skipTmpts: " + nSkipTmpts);
		Npairsj.output.println("Skip tmpts: ");
		NpairsjIO.print(getSkipTmpts());
		Npairsj.output.println("Length nTmptsPerFile: " + getNTmptsPerFile().length);
		Npairsj.output.println("No. tmpts per data file: ");
		NpairsjIO.print(getNTmptsPerFile());		
		//		Npairsj.output.println("Length scanLabels: " + volLabels.length);
		//		Npairsj.output.println("Volume labels: ");
		//		NpairsjIO.print(volLabels);
		Npairsj.output.println("Length groupLabels: " + getGroupLabels().length);
		Npairsj.output.println("Group labels: ");
		NpairsjIO.print(getGroupLabels());
		Npairsj.output.println("Length splitObjectLabels: " + splitObjectLabels.length);
		Npairsj.output.println("Split object labels: ");
		NpairsjIO.print(splitObjectLabels);

		Npairsj.output.println("Sum (length of skipTmpts + numVols): " + (numVols+nSkipTmpts));
		Npairsj.output.println("Init feat select? " + initFeatSelect); 
		Npairsj.output.println("Init svd? " + initEVD);
		Npairsj.output.println("Data reduction factor: " + dataReductionFactor);
		Npairsj.output.println("Resample data? " + resampleData);
		Npairsj.output.println("Switch train and test sets? " + switchTrainAndTestSets);
		Npairsj.output.println("Num splits (upper bound): " + numSplits);
		Npairsj.output.println("Num obj in each split: [" + numSplitObjInSplits[0] + ", " 
				+ numSplitObjInSplits[1] + "]");



		Npairsj.output.println("Run full data analysis? " + runFullDataAnalysis);
		Npairsj.output.println("Preprocess data? " + preProcess);
		Npairsj.output.println("Run glm? " + glmRun);
		Npairsj.output.println("Run pca? " + pcaRun);
		if (pcaRun) {
			Npairsj.output.println("Save pca results? " + savePCAResults);
			//			if (savePCAResults) {			
			//				Npairsj.output.println("Num dims to examine in PCA: " + pcaNumDimsToExamine);
			//			}
		}
		Npairsj.output.println("Run cva? " + cvaRun);
		if (cvaRun) {
			Npairsj.output.println("Save cva results? " + saveCVAResults);
			if (saveCVAResults) {
			}
			Npairsj.output.println("Cva class labels: ");
			for (int c : getClassLabels()) {
				Npairsj.output.print(c + " ");
			}
			Npairsj.output.println();
			Npairsj.output.println("CVA pc set 1: ");
			for (int c : cvaPCSet1) {
				Npairsj.output.print(c + " ");
			}
			Npairsj.output.println();
			Npairsj.output.println("CVA pc set 2: ");
			for (int c : cvaPCSet2) {
				Npairsj.output.print(c + " ");
			}
			Npairsj.output.println();
			Npairsj.output.println("CVA pc set ALL: ");
			for (int c : cvaPCSetAll) {
				Npairsj.output.print(c + " ");
			}
			Npairsj.output.println();
		}
	}


	public boolean equals(NpairsjSetupParams nsp) {

		if(loadDatamats) { 
			String[] datamatFilenames = getDatamatFilenames();
			if (datamatFilenames.length != nsp.getDatamatFilenames().length) {
				return false;
			}
			for (int i = 0; i < datamatFilenames.length; ++i) {
				if (!datamatFilenames[i].equals(nsp.getDatamatFilenames()[i])) {
					return false;
				}
			}
		}
		else { // block design --> no datamats, just data files
			if (nDataFiles != nsp.nDataFiles) {
				return false;

			}
			for (int i = 0; i < nDataFiles; ++i) {
				if (!getDataFilenames()[i].equals(nsp.getDataFilenames()[i])) {
					return false;
				}	
			}		
		}
		//		if (dataIs4D != nsp.dataIs4D) {
		//			return false;
		//		}
		if (!Arrays.equals(getSkipTmpts(), nsp.getSkipTmpts())) {
			return false;
		}	
		if (!Arrays.equals(getNTmptsPerFile(), nsp.getNTmptsPerFile())) {
			return false;
		}

		//		if (!eventRelAnalysis) {
		//			if (nMaskFiles != nsp.nMaskFiles) {
		//				return false;
		//			}
		//			for (int i = 0; i < maskFileNames.length; ++i) {
		//				if (!maskFileNames[i].equals(nsp.maskFileNames[i])) {
		//					return false;
		//				}
		//			}
		//		}

		if (numVols != nsp.numVols) {
			return false;
		}	
		//		if (!Arrays.equals(volDims3D, nsp.volDims3D)) {
		//			return false;
		//		}
		//		if (!Arrays.equals(volLabels, nsp.volLabels)) {
		//			return false;
		//		}	
		// **Data reduction options**	
		if (initFeatSelect != nsp.initFeatSelect) {
			return false;
		}		
		if (initEVD != nsp.initEVD) {
			return false;
		}	
		if (normedEVD != nsp.normedEVD) {
			return false;
		}
		if (loadEVD != nsp.loadEVD) {
			return false;
		}
		if (!evdFilePref.equals(nsp.evdFilePref)) {
			return false;
		}		
		if (dataReductionFactor != nsp.dataReductionFactor) {
			return false;
		}	
		// **Resampling options**		
		if (resampleData != nsp.resampleData) {
			return false;
		}		
		if (splits == null) {
			if (nsp.splits != null) {
				return false;
			}
		}
		else {
			if (nsp.splits == null) {
				return false;
			}
			else {
				if (splits.length != nsp.splits.length) {
					return false;
				}
				for (int i = 0; i < splits.length; ++i) {
					if (splits[i].length != nsp.splits[i].length) {
						return false;
					}
					for (int j = 0; j < splits[i].length; ++j) {
						if (!Arrays.equals(splits[i][j], nsp.splits[i][j])) {
							return false;
						}
					}
				}
			}
		}


		if (switchTrainAndTestSets != nsp.switchTrainAndTestSets) {
			return false;
		}
		if (numSplits != nsp.numSplits) {
			return false;
		}	
		if (!splitsInfoFilename.equals(nsp.splitsInfoFilename)) {
			return false;
		}
		if (!Arrays.equals(numSplitObjInSplits, nsp.numSplitObjInSplits)) {
			return false;
		}		
		if (!Arrays.equals(getGroupLabels(), nsp.getGroupLabels())) {
			return false;
		}		
		if (!Arrays.equals(splitObjectLabels, nsp.splitObjectLabels)) {
			return false;
		}	
		if (!Arrays.equals(subjectLabels, nsp.subjectLabels)) {
			return false;
		}		
		if (!Arrays.equals(getSessLabels(), nsp.getSessLabels())) {
			return false;
		}		
		if (runFullDataAnalysis != nsp.runFullDataAnalysis) {
			return false;
		}
		if (saveFullDataAnalysis != nsp.saveFullDataAnalysis) {
			return false;
		}			
		if (saveSplitDataResults != nsp.saveSplitDataResults) {
			return false;
		}		
		// ** Data preprocessing options**		
		if (preProcess != nsp.preProcess) {
			return false;
		}
		// + preproc vars:		
		if (removeSessionMeans != nsp.removeSessionMeans) {
			return false;
		}	
		// **GLM options (not currently implemented)**
		if (glmRun != nsp.glmRun) {
			return false;
		}	
		// **PCA options**		
		if (pcaRun != nsp.pcaRun) {
			return false;
		}	
		if (pcEigimsToBigSpace != nsp.pcEigimsToBigSpace) {
			return false;
		}		
		if (normalizePCsBySD != nsp.normalizePCsBySD) {
			return false;
		}		
		if (savePCAResults != nsp.savePCAResults) {
			return false;
		}		
		// **CVA options**
		if (cvaRun != nsp.cvaRun) {
			return false;
		}		
		if (saveCVAResults != nsp.saveCVAResults) {
			return false;
		}
		if (!Arrays.equals(getClassLabels(), nsp.getClassLabels())) {
			return false;
		}		
		if (numNPAIRS != nsp.numNPAIRS) {
			return false;
		}		
		if (pcMultFactor != nsp.pcMultFactor) {
			return false;
		}		
		if (pcStep != nsp.pcStep) {
			return false;
		}	
		if (minPCsSplit != nsp.minPCsSplit) {
			return false;
		}		
		if (minPCsFull != nsp.minPCsFull) {
			return false;
		}	    
		if (!Arrays.equals(cvaPCSet1, nsp.cvaPCSet1)) {
			return false;
		}		
		if (!Arrays.equals(cvaPCSet2, nsp.cvaPCSet2)) {
			return false;
		}	
		if (!Arrays.equals(cvaPCSetAll, nsp.cvaPCSetAll)) {
			return false;
		}		
		if (useCondsAsClasses != nsp.useCondsAsClasses) {
			return false;
		}
		// Results file options
		if (!resultsFilePrefix.equals(nsp.resultsFilePrefix)) {
			return false;
		}		
		if (!baseResultsFilePrefix.equals(nsp.baseResultsFilePrefix)) {
			return false;
		}		
		if (saveLotsOfFiles != nsp.saveLotsOfFiles) {
			return false;
		}

		return true;
	}


	/** Sets loadEVD to input arg value.
	 * 
	 */
	public void setEVDLoad(boolean loadEVD) {
		this.loadEVD = loadEVD;
	}

	/** Sets EVD file prefix to results file prefix (if not already set)
	 */
	public void setEVDFilePrefix() {
		if (evdFilePref.isEmpty()) {
			evdFilePref = resultsFilePrefix;
		}
	}

	public String[] getDataFilenames() {
		String[] dataFilenames = new String[nDataFiles];
		int count = 0;
		for (int i = 0; i < sessInfo.length; ++i) {
			Iterator<String> iter = sessInfo[i].getDataFilenames().iterator();
			while (iter.hasNext()) {
				dataFilenames[count] = iter.next();
				++count;
			}
		}
		return dataFilenames;
	}

	public int[] getNTmptsPerFile() {
		int[] nTmptsPerFile = new int[nDataFiles];
		int count = 0;
		for (int i = 0; i < sessInfo.length; ++i) {
			Iterator<Integer> iter = sessInfo[i].getNTmptsPerFile().iterator();
			while (iter.hasNext()) {
				nTmptsPerFile[count] = iter.next();
				++count;
			}
		}
		return nTmptsPerFile;
	}


	public int[] getSkipTmpts() {
		int[] skipTmpts = new int[nSkipTmpts];
		int count = 0;
		int tmptOffset = 0;
		for (int i = 0; i < sessInfo.length; ++i) {
			Iterator<Integer> iter = sessInfo[i].getSkipTmpts().iterator();
			while (iter.hasNext()) {
				skipTmpts[count] = iter.next() + tmptOffset;
				++count;
			}
			tmptOffset +=  sessInfo[i].getNumTotalVols();
		}
		return skipTmpts;

	}

	public int getNSkipTmpts() {
		return nSkipTmpts;
	}

	public String[] getMaskFilenames() {
		String[] maskFilenames = new String[sessInfo.length];
		for (int i = 0; i < sessInfo.length; ++i) {
			maskFilenames[i] = sessInfo[i].getMaskFilename();
		}
		return maskFilenames;
	}

	public String[] getDatamatFilenames() {
		String[] datamatFilenames = new String[sessInfo.length];
		for (int i = 0; i < sessInfo.length; ++i) {
			datamatFilenames[i] = sessInfo[i].getDatamatFilename();
		}
		return datamatFilenames;
	}

	public boolean[] inclAllVoxels() {
		boolean[] inclAllVox = new boolean[sessInfo.length];
		for (int i = 0; i < sessInfo.length; ++i) {
			inclAllVox[i] = sessInfo[i].inclAllVoxels();
		}
		return inclAllVox;
	}

	public double[] getMaskThreshVals() {
		double[] maskThreshVals = new double[sessInfo.length];
		for (int i = 0; i < sessInfo.length; ++i) {
			maskThreshVals[i] = sessInfo[i].getMaskThreshVal();
		}
		return maskThreshVals;
	}

	public int[] getGroupLabels() {
		return groupLabels;
	}

	public int[] getSessLabels() {
		return sessionLabels;
	}

	// Assumed: subject and session are equivalent
	public int[] getSubjLabels() {
		return subjectLabels;
	}
	
	public boolean useEigimsInProcrust() {
		return calcProcWithEigims;
	}

}


// **************************************************************************************

final class NpairsSetupSessInfo {

	private boolean debug = false;

	//	private int [] numSplitObjInSplits = new int[2];

	private int[] conditionSelection;

	private int nDataFiles = 0;

	private int nSkipTmpts = 0;

	private int numVols = 0;

	private boolean useCondsAsClasses = true;

	/** Contains info about each session file in current analysis
	 * 
	 */
	private Vector<NpairsSessionInfo> vSessionInfo;

	private Vector<Integer> vGroupLabels;

	boolean blockSessFiles = false;

	private Vector<Integer> vSessLabels;


	/*  If eventRelAnalysis = true, data is loaded from datamats instead of from input image files.
     Scans from each condition/class are averaged within subject in the datamats, hence the total
     number of data rows will not be the number of input volumes.  All corresponding label arrays
     (e.g. sessionLabels, cvaClassLabels, groupLabels) must be constructed differently for datamats
     than when data is loaded directly from image files.*/
	protected NpairsSetupSessInfo(MLStructure npairsSetupParamStruct, boolean eventRelAnalysis) 
	throws NpairsjException {

		setSessFileInfo(npairsSetupParamStruct, eventRelAnalysis);
	}

	public int[] getCondSelection() {
		return conditionSelection;
	}

	// Assumed: subject and session are equivalent
	protected int[] getSubjLabels() {
		return getSessLabels();
	}

	protected int[] getSessLabels() {
		int[] sessionLabels = new int[numVols];
		int count = 0;
		for (int i = 0; i < vSessionInfo.size(); ++i) {
			for (int j = 0; j < vSessionInfo.get(i).getNumVols(); ++j) {
				sessionLabels[count] = vSessLabels.get(i);
				++count;
			}	
		}
		return sessionLabels;
	}

	protected int[] getGroupLabels() {
		int[] groupLabels = new int[numVols];
		int count = 0;
		for (int i = 0; i < vSessionInfo.size(); ++i) {
			for (int j = 0; j < vSessionInfo.get(i).getNumVols(); ++j) {
				groupLabels[count] = vGroupLabels.get(i);
				++count;
			}	
		}
		return groupLabels;
	}

	int[] getRunLabels() {
		int[] runLabels = new int[numVols];
		int count = 0;
		int sessOffset = 0;
		for (int i = 0; i < vSessionInfo.size(); ++i) {
			Iterator<Integer> iter = vSessionInfo.get(i).getRunLabels().iterator();
			while (iter.hasNext()) {
				runLabels[count] = iter.next() + sessOffset;
				++count;			
			}
			int currInclRuns = vSessionInfo.get(i).getNumInclRuns();
			sessOffset += currInclRuns;
		}
		return runLabels;
	}

	public int[] getClassLabels() {
		int[] classLabels = new int[numVols];
		int count = 0;
		for (int i = 0; i < vSessionInfo.size(); ++i) {
			Iterator<Integer> iter = vSessionInfo.get(i).getClassLabels().iterator();
			while (iter.hasNext()) {
				classLabels[count] = iter.next();
				++count;
			}
		}
		return classLabels;
	}

	protected int getNSkipTmpts() {
		return nSkipTmpts;
	}

	protected int getNDataFiles() {
		return nDataFiles;
	}

	protected int getNumVols() {
		return numVols;
	}
	
	protected boolean useCondsAsClasses() {
		return useCondsAsClasses;
	}

	private void setSessFileInfo(MLStructure npairsSetupParamStruct, boolean eventRelAnalysis)
	throws NpairsjException {

		Hashtable<String, Integer> classLabelMap = new Hashtable<String, Integer>();		
		MLStructure sessFileInfo = (MLStructure)npairsSetupParamStruct.
		getField("session_file_info");
		int numGrps = sessFileInfo.getN();
		int totalNumSess = 0;

		// which classes skipped?
		MLDouble classSelInfo = (MLDouble)npairsSetupParamStruct.getField("class_selection");
		int numClasses = classSelInfo.getM();
		conditionSelection = new int[numClasses];
		for (int i = 0; i < numClasses; ++i) {
			conditionSelection[i] = classSelInfo.get(i).intValue();
		}
		
		// try loading class info from file
		String classFileName = "";
		try {
			classFileName = ((MLChar)npairsSetupParamStruct.getField("cva_class_file")).getString(0);
		}
		catch (NullPointerException e) { // no class file loaded
		}

		if (classFileName.length() > 0) {
			if (eventRelAnalysis) {
				throw new NpairsjException("Cannot load class information using a file for " +
				"event-related NPAIRS");
			}
			useCondsAsClasses = false;
		}	
		int[] condLabelsFromFile = null; 
		if (!useCondsAsClasses) { // get condition labels from input class file instead

			// TODO: classfile(s) should contain separate class label info for each run of each 
			// session file.
			// Currently assume all runs are of same length and that class labelling
			// provided in file applies to all runs.
			try {
				condLabelsFromFile = NpairsjIO.readIntsFromFile(classFileName);
			}
			catch (IOException x) {
				throw new NpairsjException("Could not read class file " +
						classFileName);
			}
		}
		else { // get condition labels from first session file if using conds as classes
			classLabelMap = getClassLabelMap(sessFileInfo, numClasses);		
		}

		vSessionInfo = new Vector<NpairsSessionInfo>();
		vGroupLabels = new Vector<Integer>();
		vSessLabels = new Vector<Integer>();

		for (int grp = 0; grp < numGrps; ++grp) {

			MLCell cCurrSessFiles = (MLCell)sessFileInfo.getField("session_files", 0, grp);
			int numSessFiles = cCurrSessFiles.getN();
			MLCell cCurrIgnRuns = (MLCell)sessFileInfo.getField("ignore_runs", 0, grp);

			System.out.println("Alan: numSessFiles = " + numSessFiles);
			
			for (int sf = 0; sf < numSessFiles; ++sf) {	
				++totalNumSess;
				String currSessFile = ((MLChar)cCurrSessFiles.get(sf, 0)).getString(0);

				if (totalNumSess == 1) { 
					blockSessFiles = currSessFile.endsWith(BfMRISessionFileFilter.EXTENSION);
				}				

				// which runs included?
				int[] inclRuns = null;
				try {
					// first read in which runs excluded from setup info, if available
					int[] exclRuns = null;
					String[] runExclInfo = ((MLChar)cCurrIgnRuns.get(sf, 0)).getString(0).split("\\s");
					int numExclRuns = runExclInfo.length;
					for (String s : runExclInfo) {
						if (s.equals("0")) --numExclRuns; // don't count placeholder '0' as excluded run
						if (s.equals("")) --numExclRuns; // don't count empty string as excluded run
					}
					if (debug) {
						System.out.println("How many runs excl? " + numExclRuns);
					}
					exclRuns = new int[runExclInfo.length];
					for (int i = 0; i < numExclRuns; ++i) {
						exclRuns[i] = Integer.parseInt(runExclInfo[i]);
					}
					if (debug) {
						System.out.println("Excl runs for sess " + sf + ", group " + grp + ": ");
						NpairsjIO.print(exclRuns);
					}
                    
					// now determine which runs included
					MLStructure currSessFileInfo = (MLStructure)new NewMatFileReader(currSessFile).
							getContent().get("session_info");
					int currNumRuns = ((MLDouble)currSessFileInfo.getField("num_runs")).getIntArray()[0][0];
					inclRuns = new int[currNumRuns - numExclRuns];
					int c = 0;
					for (int i = 1; i <= currNumRuns; ++i) {
						if (!MLFuncs.contains(exclRuns, i)) {
							inclRuns[c] = i;
							++c;
						}
					}
				}
				catch (IOException ioe) {
					System.out.println("Unable to read session file: " + currSessFile);
					return;
				}
				catch (NullPointerException npe) {
					// Run incl. info not contained in setup file; will be read in
					// from session files instead
					if (debug) {
						System.out.println("Run incl. info not contained in setup file; will be read in from session files instead");
					}
				}
				
				if (debug) {
					System.out.println("Incl runs: ");
					try {
					NpairsjIO.print(inclRuns);
					}
					catch (NullPointerException npe) {
						System.out.println("null");
					}
				}
				NpairsSessionInfo currInfo = new NpairsSessionInfo(currSessFile, classLabelMap, 
						condLabelsFromFile, conditionSelection, eventRelAnalysis, inclRuns);
				vSessionInfo.add(currInfo);
				vGroupLabels.add(grp);
				vSessLabels.add(totalNumSess); // totalNumSess incremented for every sf
				nDataFiles += currInfo.getNDataFiles();
				nSkipTmpts += currInfo.getNSkipTmpts();	
				numVols += currInfo.getNumVols();
			} 		
		}		
	}

	/** Reads condition info from given sessFileInfo structure and returns map containing
	 *  condition names as keys and (Integer) class labels as values.
	 * @param sessFileInfo
	 * @param numClasses
	 * @return classLabelMap 
	 * @throws NpairsjException
	 */
	private Hashtable<String, Integer> getClassLabelMap(MLStructure sessFileInfo, 
			int numClasses) throws NpairsjException {
		String firstSessFile = ((MLChar)((MLCell)sessFileInfo.getField("session_files", 0, 0)).
				get(0, 0)).getString(0);
		MLStructure firstSessFileInfo = null;
		try {
			firstSessFileInfo = (MLStructure)new NewMatFileReader(firstSessFile).
			getContent().get("session_info");
		} 
		catch (Exception ex) {
			throw new NpairsjException("Session file " + firstSessFile 
					+ " could not be loaded.");

		}		
		String[] condNames = MLFuncs.MLCell1dRow2StrArray
		((MLCell)firstSessFileInfo.getField("condition"));
		if (condNames.length != numClasses) {
			throw new NpairsjException("Number of conditions in first session file: " + condNames.length + 
					".\nNumber of conditions expected in setup file: " + numClasses + ".");
		}

		int nextClassLabel = 1;
		Hashtable<String, Integer> classLabelMap = new Hashtable<String, Integer>(numClasses);
		for (int c = 0; c < condNames.length; ++c) {
			if (!classLabelMap.containsKey(condNames[c])) {
				classLabelMap.put(condNames[c], 
						new Integer(nextClassLabel));
				++nextClassLabel;
			}
		}		
		return classLabelMap;
	}

	protected NpairsSessionInfo[] getSessionInfo() {
		NpairsSessionInfo[] nsi = new NpairsSessionInfo[vSessionInfo.size()];
		for (int i = 0; i < vSessionInfo.size(); ++i) {
			nsi[i] = vSessionInfo.get(i);
		}
		return nsi;
	}
}

