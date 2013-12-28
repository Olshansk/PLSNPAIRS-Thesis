package pls;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;

import pls.shared.MLFuncs;

import npairs.NpairsjException;
import npairs.NpairsjSetupParams;
import npairs.io.NpairsjIO;

public class RunAnalysisInfo {

	/** Extracts information required to run PLS or NPAIRS via command line.
	 */
	private boolean debug = false;

	private int[] numAnalyses = null; // array (length no. of input setup file) indicating
	                                  // number of analyses per input setup file
	private boolean[] loadEVD = null; // array (length no. of input setup files) indicating whether EVD
	                                  // is to be loaded for corresponding setup file (if false then
	                                  // EVD (if done) is to be run instead of loaded)
//	private boolean evdOnly = false;  // true if, for each setup file, only init. EVD is to be run (and
	                                  // saved; and then program is to exit without running actual analysis)
//	private int analysisNum = -1;     // if set, number indicates which analysis no. to execute in each input
	                                  // file
//	private boolean analysisNumSet = false; // true if analysisNum set 

	private boolean validInput = false; // true if all input info is valid

	private String[] setupFilenames; 
	
	private int numParamFiles;
	
	private int minNumAnalyses;
	
//  private boolean isNpairs = true;
    
	
	public RunAnalysisInfo(String[] argv) throws IOException, NpairsjException {
		
		numParamFiles = 0;
		for (int i = 2; i < argv.length; ++i) {
			if (argv[i].endsWith(".mat")) {
				++numParamFiles;
			}
//			else if ((argv.length <= 3) || (i < argv.length - 1)) {
//				return;
//			}
			else {
				return;
			}
//			else if (isNpairs) {
//			boolean isInt = true;
//			try {
//				analysisNum = Integer.parseInt(argv[i]);
//				analysisNumSet = true;
//			}
//			catch (NumberFormatException nfe) {
//				isInt = false;
//			}
//
//			if (argv[i].equalsIgnoreCase("EVD_ONLY")) {
//				evdOnly = true;
//			}
//			else if (!isInt) {		
//				return;
//			}
		}
//		}

		if (debug) {
			System.out.println("Checking num analyses and evd info...");
		}
		
		checkNumAnalysesAndEVDInfo(argv, numParamFiles);
//		if (analysisNumSet && !validAnalysisNum()) {
		minNumAnalyses = MLFuncs.min(numAnalyses);
//			System.out.println("Invalid analysis number entered: " + analysisNum +
//					". Must be between 1 and " + minNumAnalyses + ".");
//			return;	
//		}
		
		if (debug) {
			System.out.println("After setting: ");
			System.out.print("Num analyses: ");
			NpairsjIO.print(numAnalyses);
			System.out.print("LoadEVD? ");
			try {
				NpairsjIO.print(loadEVD);
			} catch (NullPointerException npe) {
				System.out.println("null");
			}

		}

		validInput = true;	
	}


	protected boolean checkValid(int analysisNum) {
		boolean isValid = true;
		for (int n: numAnalyses) {
			if (!(analysisNum > 0 && analysisNum <= n)) {
				isValid = false;
				break;
			}
		}
		return isValid;
	}


	private void checkNumAnalysesAndEVDInfo(String[] argv, int numParamFiles) throws 
		IOException, NpairsjException {
//		if (!evdOnly) {
			loadEVD = new boolean[numParamFiles];
			numAnalyses = new int[numParamFiles];
			setupFilenames = new String[numParamFiles];
			for (int j = 0; j < numParamFiles; ++j) {
				boolean loadDatamats = false; 
				setupFilenames[j] = argv[j + 2];
				if(setupFilenames[j].endsWith("SetupER.mat")) { // 'event-related' NPAIRS
					loadDatamats = true;
				}

				NpairsjSetupParams nsp = new NpairsjSetupParams(setupFilenames[j], loadDatamats);
				loadEVD[j] = nsp.loadEVD;
				numAnalyses[j] = nsp.numNPAIRS;
				
			}
//		}
//		else { // Each PLS setup param file can only contain a single analysis.
			// If only EVD is to be run for NPAIRS analyses, each setup param file will
			// also only be run a single time.
//			numAnalyses = new int[numParamFiles];
//			for (int n = 0; n < numParamFiles; ++n) {
//				numAnalyses[n] = 1;
//			}
//		}
	}
	

	protected int[] numAnalyses() {
		return numAnalyses;
	}

	protected boolean[] loadEVD() {
		return loadEVD;
	}

//	public boolean runEVDOnly() {
//		return evdOnly;
//	}

//	public int analysisNum() {
//		return analysisNum;
//	}
	
	protected int getMinNumAnalyses() {
		return minNumAnalyses;
	}

	protected boolean inputValid() {
		return validInput;

	}
	
	/** Saves evd/num analyses info in textfile(s) in directory containing setup file. 
	 *  Filename for each textfile will
	 *  be of the form setupFilePrefix.runAnalysisInfo.txt.
	 *  (Where setupFilePrefix is the file prefix (in front of _NPAIRSJAnalysisSetup.mat or
	 *  _NPAIRSJAnalysisSetupER.mat), including file path.)
	
	 *  NOTE: files are overwritten if they already exist.
	 *  @throws IOException if a file cannot be created and/or written to
	 * 
	 */
	protected void save() throws IOException {
		for (int i = 0; i < numParamFiles; ++i) {
			String setupFilePref = getSetupFilePrefix(setupFilenames[i]);

		//	String saveDir = System.getProperty("user.dir");
			File f = new File(setupFilePref + ".npairsRunInfo.txt");
			
            String loadOrCreateEVD = "loadIt";
            if (!loadEVD[i]) {
            	loadOrCreateEVD = "createIt";
            }  
            
            try {
            	PrintStream output = new PrintStream(f);
            	output.println("EVD = " + loadOrCreateEVD);
            	output.println("NUM_ANALYSES = " + numAnalyses[i]);
            	output.close();		
            } catch (IOException ioe) {
            	throw new IOException("Could not create file " + f.getAbsolutePath());
            }
		}
	}
	


	/** prefix include full path to setup file directory.
	 * 
	 * @param setupFilename
	 * @return prefix including path
	 */
	private String getSetupFilePrefix(String setupFilename) {
		String filenameInclPath = (new File(setupFilename).getAbsolutePath());
		int lastIdx = filenameInclPath.indexOf("_NPAIRSAnalysisSetup");
		String filePref = filenameInclPath.substring(0, lastIdx);
		return filePref; 
	}
}
