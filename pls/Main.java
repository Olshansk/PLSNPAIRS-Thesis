package pls;

import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Vector;

import javax.swing.JOptionPane;

import npairs.Npairsj;
import npairs.NpairsjSetupParams;
import npairs.io.NpairsjIO;
import pls.analysis.Analysis;
import pls.analysis.NpairsAnalysis;
import pls.chrome.MainFrame;
import pls.chrome.result.LoadedVolumesDialog;
import pls.chrome.result.PlsResultLoader;
import pls.chrome.result.ResultFrame2;
import pls.sessionprofile.RunGenerateDatamat;
import pls.sessionprofile.RunInformation;
import pls.sessionprofile.SessionProfile;
import pls.shared.BfMRIResultFileFilter;
import pls.shared.MLFuncs;
import pls.shared.PlsAnalysisSetupFileFilter;
import pls.shared.StreamedProgressHelper;
import pls.shared.fMRIResultFileFilter;

import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLStructure;

import extern.NewMatFileReader;

/**
 * Main class that drives the PLSNPAIRS program
 * @author imran
 *
 */
public class Main {
	/**
	 * Main entry point into the program:
	 * If one or more filenames are given as argument, they are treated as batch files:
	 * <ul>
	 * 	<li>The files should contain the input for the program in a specified format.
	 * 	<li>TODO: format instructions</li>
	 * </ul>
	 * If no argument is given:
	 * <ul>
	 *   <li>A gui pops up which can be used to:
	 * 		<ul>
	 *		  <li>Create a PLS session file</li>
	 *  	  <li>Run a PLS analysis</li>
	 *  	  <li>Output the results of an analysis</li>
	 *  	</ul>
	 *   </li>
	 * </ul>
	 * @param argv Either null or command-line args and/or string filenames for parameter files.
	 * @throws Exception
	 */
	
	final static boolean debug = false;
	
	// note: run with VM arguments -Xms64m -Xmx512m
	public static void main(String[] argv) throws Exception {
		if(argv.length == 0) {
			// The java version is only really important if you're running the results GUI.
			String javaVersion = System.getProperty("java.version");
			
			double slam = Double.parseDouble(javaVersion.substring(0, javaVersion.indexOf('.', javaVersion.indexOf('.') + 1) ) );
			
			if (slam < 1.6) {
				JOptionPane.showMessageDialog(null, "Sorry, but Java version 1.6 or higher is required " +
						"to run this application.\nWe detect that you have version " + javaVersion);
			}
			
			boolean loggingEnabled = false; //change to true if you want error logging
			if (loggingEnabled) {
				MainFrame mf = new MainFrame();
				String now = getDate();
				createErrorLog(mf, now);
				mf.setNow(now);
				mf.setLoggingEnabled(true);
			} else new MainFrame();

		} else {
			runFromCommandLine(argv);
		}
				
	}

	private static void createErrorLog(MainFrame mf, String now) {
		//redirect stdOut to create a log file
		try {
			String s = System.getProperty("file.separator");
			int logNum = getAndIncrementLogNumber();
			File logFile = new File("error_logs" + s + now + "_plsnpairs_error_log" + logNum + ".txt");
			logFile.createNewFile();
			mf.outputLogStream = new PrintStream(new BufferedOutputStream(new FileOutputStream(logFile))); 
			//System.setOut(outputLog); 
			System.setErr(mf.outputLogStream);
			System.out.println("PLSNPAIRS error log" + logNum + " was created at time " + NpairsAnalysis.getDateTime());
		} catch (Exception e) {
			System.out.println("Main class threw an exception when trying to create the error log");
			e.printStackTrace();
		}
	}
	
	private static String getDate() {
		//get current date
		final String DATE_FORMAT = "yyyy-MM-dd";
		Calendar cal = Calendar.getInstance();
		SimpleDateFormat sdf = new SimpleDateFormat(DATE_FORMAT);
		String now = sdf.format(cal.getTime());
		return now;
	}
	
	private static int getAndIncrementLogNumber() throws IOException {
		//get log number
		String s = System.getProperty("file.separator");
		File logFileTracker = new File("error_logs" + s + "." + getDate() + "_log_tracker.txt");
		System.out.println(logFileTracker.getAbsolutePath()); 
		logFileTracker.createNewFile();
		BufferedReader bufRead = new BufferedReader(new FileReader(logFileTracker));
		String line = bufRead.readLine();
		int logNumber;
		if (line == null) logNumber = 1;
		else logNumber = Integer.parseInt(line);
		
		//increment log number
		PrintWriter printWrite = new PrintWriter(new BufferedWriter(new FileWriter(logFileTracker)));
		int nextLogNum = logNumber + 1;
		printWrite.write("" + nextLogNum);
		printWrite.close();
		
		return logNumber;
	}
	
	/**
	 * Load multiple result files to be displayed in the result viewer.
	 * @param argv command line arguments passed in by the user. This should 
	 * be of the form {'BATCH','file1',file2'...'filen'}
	 */
	private static void batchLoad(String[] argv){
		List<String> files = new LinkedList<String>();
		String lastPath;
		
		final MainFrame mFrame;
		ResultFrame2 result;
		
		try {
			mFrame = new MainFrame();
		} catch (Exception e) {
			e.printStackTrace();
			return;
		}
		
		for(int i = 1; i < argv.length; i++){
			File thisFile = new File(argv[i]);
			
			if(thisFile.canRead()) files.add(argv[i]);
			else System.err.print("Could not load: " + argv[i]);
		}
		lastPath = ((LinkedList<String>) files).getLast();
				
		LoadedVolumesDialog.setLastPath(PlsResultLoader.getPrefix(lastPath));

		result = new ResultFrame2("Batch", files);
		// Disables the button to open this window after it has
		// been opened. We only want one results window to be
		// opened by one instance of this application.
		result.addWindowListener(new WindowAdapter() {
			
			@Override
			public void windowOpened(WindowEvent e) {
				mFrame.getResultsButton().setEnabled(false);
			}
			
			@Override
			public void windowClosed(WindowEvent e) {
				mFrame.getResultsButton().setEnabled(true);
				((ResultFrame2)e.getSource()).removeWindowListener(this);
			}
			
		});
		
	}
	
	private static void runFromCommandLine(String[] argv) throws Exception {		
		// type parameter can be one of PLS or NPAIRS
		String type = argv[0];
		
		//Prepare to load multiple result files for the result viewer.
		if(type.equals("BATCH")){
			//for(String s : argv) System.out.println(s);
			batchLoad(argv);
			return;
		}
		
		if (!type.equals("PLS") && !type.equals("NPAIRS") ) {
			System.out.println("The first (type) parameter must be one of PLS or NPAIRS.");
			return;
		}
		
		boolean isNpairs = type.equals("NPAIRS");
		
		// Action parameter must be present
		if (argv.length < 2) {
			System.out.println("You need to specify the action you want to perform.");
			return;
		}
		
		// Get the action to perform
		String action = argv[1];
		
		// Handle SessionProfile creation
		if (action.equals("SESSIONPROFILE") ) {
			if (argv.length <= 2 || argv[2].equals("help") ) {
				printSessionProfileHelpMessage();
				return;
			}
				
			saveSessionProfile(argv, isNpairs);
		} else if (action.equals("DATAMAT") ) {
			if (argv.length <= 2 || argv[2].equals("help") ) {
				printDatamatHelpMessage();
				return;
			}
			
			createDatamat(argv);
		} else if (action.equals("SETUP_ANALYSIS") ) {
			if (argv.length <= 2 || argv[2].equals("help") ) {
				if (isNpairs) {
					printNpairsSetupAnalysisHelpMessage();
				} else {
					printPlsSetupAnalysisHelpMessage();
				}
				return;
			}
			if (isNpairs) {
				setupNpairsAnalysis(argv);
			} else {
				setupPlsAnalysis(argv);
			}
		} else if (action.equals("RUN_ANALYSIS") ) {
			if (argv.length <= 2 || argv[2].equals("help") ) {
				printRunAnalysisHelpMessage();
				return;
			}
			
			// If last arg is int, make sure it's a valid analysis num
			try {
				int analysisNum = Integer.parseInt(argv[argv.length - 1]);
				String[] paramFileList = new String[argv.length - 1];
				System.arraycopy(argv, 0, paramFileList, 0, argv.length - 1);
				RunAnalysisInfo info = new RunAnalysisInfo(paramFileList);
				boolean validNum = info.checkValid(analysisNum);
				if (!validNum) {
					System.out.println("Invalid analysis number entered: " + analysisNum +
					". Must be between 1 and " + info.getMinNumAnalyses() + ".");
					return;
				}
			} 
			catch (NumberFormatException nfe) {}
			
			runAnalysis(argv, isNpairs);
			
		} else if (action.equals("CHECK_FILE") ) {
			if (argv.length <= 2 || argv[2].equals("help") ) {
				printCheckFileHelpMessage();
				return;
			}

			if (isNpairs) {
				try {
					RunAnalysisInfo info = new RunAnalysisInfo(argv);
					if (!info.inputValid()) {
						printCheckFileHelpMessage();
						return;
					}
					// save run analysis info (evd and num analyses)
//					if (debug) {
					System.out.print("Saving run analysis info into textfile(s)...");
//					}
				
					info.save();
					
//					if (debug) {
					System.out.println("[DONE]");
//					}

					if (debug) {
						System.out.println("Input args: ");
						NpairsjIO.print(argv);
						System.out.print("Npairs or PLS? "); 
						if (isNpairs) System.out.println("NPAIRS"); else System.out.println("PLS");
						try {
							System.out.println("Num analyses: ");
							NpairsjIO.print(info.numAnalyses());
						} catch (NullPointerException npe) {
							//						if (info.runEVDOnly()) {
							//							System.out.println("Doing EVD only...");
							//						}
							//						else {
							npe.printStackTrace();
							//						}
						}
						System.out.println("Load EVD? ");
						try {
							NpairsjIO.print(info.loadEVD());
						}
						catch (NullPointerException npe) {
							System.out.println("No.");
						}
						//					if (info.runEVDOnly()) {
						//						System.out.println("Running EVD only...");					
						//					}
						//					else if (info.analysisNum() == -1 && !info.runEVDOnly()) {
						//						System.out.println("Running all analyses...");
						//					}
						//					else {
						//						System.out.println("Running analysis no. " + info.analysisNum());
						//					}
					}
				} catch (IOException ioe) { 
					// couldn't load setup param file or save run analysis info textfile
					System.out.println("\nERROR: " + ioe.getMessage());
					System.exit(1);
				}
			} else {
				printCheckFileHelpMessage();
			}
		} else {
			System.out.println("The second (action) parameter must be one of SESSIONPROFILE, DATAMAT, " +
			"SETUP_ANALYSIS, RUN_ANALYSIS or (for NPAIRS only) CHECK_FILE.");
		}
	}


	static void printCheckFileHelpMessage() {
		System.out.println("Check File usage (for NPAIRS only): ");
		System.out.println("setupfile1 setupfile2...");
		System.out.println("CHECK_FILE reads each input setup file and saves a textfile " +
				"\n(in setup file directory, 1 for each setup file) containing pertinent info, namely:" +
				"\n(i) whether initial eigenvalue decomposition is to be loaded or created; and" +
				"\n(ii) how many analyses are contained in the given setup file \n(> 1 if pc range," +
				" e.g., is set). " +
				"\nREQUIRED: setup files must have unique filenames (it is not enough that they live " +
				"in different directories); user must have permission to write into directory containing " +
				"\nsetup files.");
		
	}

	private static void saveSessionProfile(String[] argv, boolean isNPAIRS) {
		Map<String, ArrayList<String> > params = getParams(argv);
		
		if (params == null) {
			return;
		}
		
		String sessionFilePrefix = "";
		if (params.containsKey("SESSION_FILE_PREFIX") && params.get("SESSION_FILE_PREFIX").size() > 0) {
			sessionFilePrefix = params.get("SESSION_FILE_PREFIX").get(0);
		} else {
			System.out.println("Must include session file prefix.");
			return;
		}
		
		String sessionFilePath = "";
		if (params.containsKey("SESSION_FILE_DIR") && params.get("SESSION_FILE_DIR").size() > 0) {
			sessionFilePath = params.get("SESSION_FILE_DIR").get(0);
			if (!sessionFilePath.endsWith(System.getProperty("file.separator"))) {
				sessionFilePath = sessionFilePath.concat(System.getProperty("file.separator"));
			}
		}
		sessionFilePrefix = sessionFilePath.concat(sessionFilePrefix);
		
		boolean isBlock = true;
		if (params.containsKey("BLOCK") && params.get("BLOCK").size() > 0) {
				isBlock = Boolean.parseBoolean(params.get("BLOCK").get(0) );
		}
		
		SessionProfile profile = new SessionProfile(); 
		boolean loadSF = false;
		if (params.containsKey("ORIG_FILE_PREFIX")) {
			if (!(params.get("ORIG_FILE_PREFIX").size() > 0)) {
				System.out.println("Must include original session file prefix to load existing session file.");
				return;
			}
			String origFilePrefix = params.get("ORIG_FILE_PREFIX").get(0);
			try {
				profile = SessionProfile.loadSessionProfile(origFilePrefix, isBlock, isNPAIRS);
				loadSF = true;
			}
			catch (IOException ioe) {
				String plsOrNpairs = "PLS";
				if (isNPAIRS) plsOrNpairs = "NPAIRS";
				String type = "event-related";
				if (isBlock) {
					type = "block";
				}
				System.out.println("Unable to load " + type + " " + plsOrNpairs + " session file (prefix: " 
						+ origFilePrefix + ").");
				return;
			}
			
		}
		
		if (params.containsKey("CHANGE_DATA_PATH")) {
			if (!(params.containsKey("ORIG_FILE_PREFIX"))) {
				System.out.println("Must include original session file prefix to change data paths.");
				return;
			}	

//			String origFilePrefix = params.get("ORIG_FILE_PREFIX").get(0);
//			try {
//				SessionProfile profile = SessionProfile.loadSessionProfile(origFilePrefix, isBlock, isNPAIRS);
			if (params.get("CHANGE_DATA_PATH").size() > 0) {
				String newDataPath = params.get("CHANGE_DATA_PATH").get(0);
				profile.changeDataPath(newDataPath);
				//profile.saveSessionProfile(sessionFilePrefix);
			}
			else {
				System.out.println("Must include new data path to change paths.");
				//	return;
			}
				
//			} catch (IOException e) {
//				String plsOrNpairs = "PLS";
//				if (isNPAIRS) plsOrNpairs = "NPAIRS";
//				String type = "event-related";
//				if (isBlock) {
//					type = "block";
//				}
//				System.out.println("Unable to load " + type + " " + plsOrNpairs + " session file (prefix: " 
//						+ origFilePrefix + ").");
//			}
		//	return;
		}
		
//		if (params.containsKey("ORIG_FILE_PREFIX")) {
//			System.out.println("Must include new data path to change paths.");
//			return;
//		}

		
		
		String description = "";
		if (params.containsKey("DESCRIPTION") ) {
			if (params.get("DESCRIPTION").size() > 0) {
				for (String s : params.get("DESCRIPTION")) {
					description = description.concat(s + " ");
				}
				description = description.trim();
				if (loadSF) {
					profile.setDescription(description);
				}
			}
		}
		
		int[] ignoreRuns = null;
		if (params.containsKey("IGNORE_RUNS") ) {
			ArrayList<String> data = params.get("IGNORE_RUNS");
			ignoreRuns = new int[data.size()];
			
			for (int i = 0; i < ignoreRuns.length; ++i) {
				try {
					ignoreRuns[i] = Integer.parseInt(data.get(i));
				} catch (NumberFormatException nfex) {
					System.out.println("Runs to ignore must be integers.");
					return;
				}
			}
			if (loadSF) {
				profile.setIgnoreRuns(ignoreRuns);
			}
		}
		
		boolean useMaskFile = false;
		String brainMaskFile = "";
		if (params.containsKey("USE_MASK_FILE")) {
			if (params.get("USE_MASK_FILE").size() == 0 ||
				    params.get("USE_MASK_FILE").get(0).equalsIgnoreCase("true")) {
				useMaskFile = true;
				if (!(params.containsKey("BRAIN_MASK_FILE") ||
						(loadSF && profile.brainMaskFile.length() > 0))) {
					System.out.println("If you want to use a brain mask, you must specify the file.");
					return;
				}					
			}
			else if (params.get("USE_MASK_FILE").get(0).equalsIgnoreCase("false")) {
				if (isBlock && isNPAIRS) {
					System.out.println("You must provide a brain mask file for NPAIRS analysis " +
							"of Block fMRI.");
					return;
				}
				useMaskFile = false;
				brainMaskFile = "";
			}
		}
		
		if (params.containsKey("BRAIN_MASK_FILE") ) {
			if (params.containsKey("USE_MASK_FILE") && useMaskFile == false) {
				System.out.println("WARNING: Ignoring -BRAIN_MASK_FILE because USE_MASK_FILE " +
						"set to false.");
			}
			else if (params.get("BRAIN_MASK_FILE").size() > 0) {
				brainMaskFile = params.get("BRAIN_MASK_FILE").get(0);
				useMaskFile = true;
				profile.setMaskFile(brainMaskFile);
			} else {
				System.out.println("If you want to use a brain mask, you must specify the file.");
				return;
			}
		}
		else if (isNPAIRS && isBlock) {
			 if(!loadSF || (loadSF && profile.brainMaskFile.length() == 0)) {
				 System.out.println("You must provide a brain mask file for NPAIRS analysis of Block fMRI.");
					return;
			 }	
		}
		
		int numSkippedScans = 0;
		if (params.containsKey("SKIP_SCANS") ) {
			if (params.get("SKIP_SCANS").size() > 0) {
				numSkippedScans = Integer.parseInt(params.get("SKIP_SCANS").get(0) );
				if (loadSF) {
					profile.setNumSkippedScans(numSkippedScans);
				}
			}
		}
		
		boolean mergeAcrossRuns = true;
		if (params.containsKey("MERGE_ACROSS_RUNS") && params.get("MERGE_ACROSS_RUNS").size() > 0) {
				mergeAcrossRuns = Boolean.parseBoolean(params.get("MERGE_ACROSS_RUNS").get(0) );
				if (loadSF) {
					profile.setMergeAcrossRuns(mergeAcrossRuns);
				}
		}
		
		Vector<String[]> conditionInfo = new Vector<String[]>();
		int nConds = 0;
		if (params.containsKey("CONDITION_INFO") ) {
			for (String condInfo : params.get("CONDITION_INFO") ) {
				++nConds;
				String[] condInfoArray = condInfo.split(",");
				if (condInfoArray.length == 3 &&
						(!condInfoArray[1].equals("0") || !condInfoArray[2].equals("1")))
					System.out.println("WARNING! Setting condition ref. scan info [" +
							condInfoArray[1] + ", " + condInfoArray[2] + "] to [0, 1].");
				
				
					///////////////////////////////////////////////////////////////////
					//////// Always set condition ref scan info to 0,1 for now,////////
					//////// since that is the only setting that currently works!//////
					///////////////////////////////////////////////////////////////////
				// if (condInfoArray.length == 1) {	
					//					if (isNPAIRS && isBlock) {
//						// don't need ref. scan info; add default ref. scan info
//						// if not included in input condition info for compatibility 
//						// with PLS session files
						condInfoArray = new String[] {condInfoArray[0],"0","1"};	
//					}
//					else {
//						System.out.println("Must include ref. scan info in CONDITION_INFO for datamat " +
//								"normalization.");
//						return;
//					}
				//}
				conditionInfo.add(condInfoArray);			
			}
			if (loadSF) {
				profile.setConditionInfo(conditionInfo);
			}
		}
		else if (!loadSF) {
			System.out.println("Must include condition information: name and (if not Block NPAIRS) " +
					"ref. scan details.");
		}
		
		Vector<RunInformation> runInfo = new Vector<RunInformation>();
		if (params.containsKey("DATA_PATHS") && params.containsKey("DATA_FILES")) {
									
				ArrayList<String> runPaths = params.get("DATA_PATHS");
				ArrayList<String> runFiles = params.get("DATA_FILES");
				
				ArrayList<String> runOnsets = null;
				// get input onsets for each run and condition
				if (params.containsKey("ONSETS")) {
					 runOnsets = params.get("ONSETS");
				}
				else if(params.containsKey("ONSET_FILES") ) {
					runOnsets = loadOnsetsFromFiles(params.get("ONSET_FILES"), isBlock, nConds);
					if (runOnsets == null) {
						return;
					}
				}
			
			ArrayList<String> runLengths = null;
			if (isBlock) {
				if (params.containsKey("BLOCK_LENGTHS")) {
					runLengths = params.get("BLOCK_LENGTHS");
				} else if (params.containsKey("ONSET_FILES")) {
					// read block lengths from onset file
					runLengths = loadLengthsFromFiles(params.get("ONSET_FILES"));
				}
				else {
					System.out.println("Must include block length for each onset.");
					return;
				}
			}

			boolean paramSizesMatch = true;
			if (runPaths.size() == runFiles.size() &&
					runPaths.size() == runOnsets.size()) {
				
				for (int i = 0; i < runPaths.size(); ++i) {
					String runPath = runPaths.get(i);
					
					ArrayList<String> onsetsList = new ArrayList<String>();
					String[] onsets = runOnsets.get(i).split(":");
					if (onsets.length != nConds) {
						System.out.println("Number of ONSETS entries must equal " +
							"number of conditions.");
						return;
					}
					
					int[] numOnsets = new int[onsets.length];
					int c = 0;
					for (String onset : onsets) {
						String[] individuals = onset.split(",");
						numOnsets[c] = individuals.length;
						++c;
						String onsetString = "";
						for (String s : individuals) {
							onsetString += s + " ";
						}
						onsetString = onsetString.trim();
						onsetsList.add(onsetString);
					}
					// Replace commas with spaces
					String runFilesString = runFiles.get(i).replaceAll(",", " ");		
					
					RunInformation ri = new RunInformation(runPath, runFilesString, onsetsList);
					
					if (isBlock) { 
						if (runPaths.size() == runLengths.size()) {

							ArrayList<String> lengthsList = new ArrayList<String>();
							String[] lengths = runLengths.get(i).split(":");
							if (lengths.length != nConds) {
								System.out.println("Number of BLOCK_LENGTHS entries must equal " +
										"number of conditions.");
								return;
							}
							int j = 0;
							for(String length : lengths) {
								String[] individuals = length.split(",");
								if (individuals.length != numOnsets[j]) {
									System.out.println("Number of block length entries for each condition " +
											"must be equal to number of onsets.");
									return;
								}
								++j;
								String lengthString = "";
								for (String s : individuals) {
									lengthString += s + " ";
								}
								lengthString = lengthString.trim();
								lengthsList.add(lengthString);
							}
							ri.lengths = lengthsList;
							
						} else { // wrong size lengths param
							paramSizesMatch = false;
						}	
					}
					
					runInfo.add(ri);
				}
			} else {
				paramSizesMatch = false;
			}
			if (!paramSizesMatch) {
				System.out.println("All pieces of run info (DATA_PATHS, DATA_FILES, ONSETS, [BLOCK_LENGTHS]) " +
						"\nmust have the same length.");
				return;
			}
			
		} else if (!loadSF) {
			System.out.println("At least one of DATA_PATHS, DATA_FILES or ONSETS is missing.");
			return;
		}
		
		String datamatPrefix = "";
		if (params.containsKey("DATAMAT_PREFIX") && 
				(params.get("DATAMAT_PREFIX").size() > 0)) {
				datamatPrefix = params.get("DATAMAT_PREFIX").get(0);
				if (loadSF) {
					profile.setDmatPrefix(datamatPrefix);
				}
		}
	    else if (!loadSF){
			datamatPrefix = new File(sessionFilePrefix).getName();
			System.out.println("Using session file name as datamat prefix: " + datamatPrefix);
		}
		
		if (!loadSF) {
			profile = new SessionProfile(description, datamatPrefix, mergeAcrossRuns, conditionInfo,
				runInfo, useMaskFile, brainMaskFile, isNPAIRS, isBlock, numSkippedScans, ignoreRuns, false);
		}
		try {
			profile.saveSessionProfile(sessionFilePrefix);
		} catch (IOException e) {
			System.out.println("Unable to save session profile" + sessionFilePrefix);
			e.printStackTrace();
		} catch (NullPointerException npe) {
			System.out.println(npe.getMessage());
		}
	}
	
	private static ArrayList<String> loadLengthsFromFiles(
			ArrayList<String> fileList) {
		ArrayList<String> lengthValues = new ArrayList<String>();
		for (int r = 0; r < fileList.size(); ++r) {
			String filename = fileList.get(r);
			String currRunLengths = "";
			try {
			BufferedReader br = new BufferedReader(new FileReader(filename));
			String line = br.readLine();
			while (line != null && line.length() > 0) {
				line = br.readLine();
				String[] lengthsLine = line.split("\\s");
				for (String o: lengthsLine) {
					currRunLengths = currRunLengths.concat(o + ",");
				}
				currRunLengths = currRunLengths.substring(0, currRunLengths.length() - 1);
				currRunLengths = currRunLengths.concat(":");
				
//				System.out.println("curr run lengths: " + currRunLengths);
				line = br.readLine();
			}
			lengthValues.add(currRunLengths);	
			}
			catch (Exception e) {
				System.out.println("Unable to load length info from onset file " + filename);
				return null;
			}
		}
		return lengthValues;
	}

	private static boolean areOnsetValuesValid(String line, String origin) {
		int prev = Integer.MIN_VALUE;
		
		String[] values = line.split(" ");
		for (int i = 0; i != values.length; i++) {
			try {
				int current = Integer.parseInt(values[i]);
				if (current == prev) {
					System.out.println(origin + " contains a duplicate onset value: " + current);
					return false;
				} else if (current < prev) {
					System.out.println(origin + " has the onset value " + current + " after the onset value " + prev + ".");
					return false;
				} else if (current < -1) {
					System.out.println(origin + " should not have any onset values less than -1.");
					return false;
				}
				
				prev = current;
			} catch (NumberFormatException e) {
				System.out.println(origin + " contains an invalid onset value: " + values[i]);
				return false;
			}
		}
		return true;
	}
	
	private static boolean areLengthValuesValid(String line, String origin) {
		String[] values = line.split(" ");
		for (int i = 0; i != values.length; i++) {
			try {
				int current = Integer.parseInt(values[i]);
				if (current < -1) {
					System.out.println(origin + " should not have any length values less than -1.");
					return false;
				}
			} catch (NumberFormatException e) {
				System.out.println(origin + " contains an invalid length value: " + values[i]);
				return false;
			}
		}
		return true;
	}

	private static boolean sameLength(String onsetsLine, String lengthsLine, String origin) {
		int numOnsets = onsetsLine.split(" ").length;
		int numLengths = lengthsLine.split(" ").length;
		
		boolean result = (numOnsets == numLengths);
		
		if (!result) {
			System.out.println(origin + " does not contain an equal number of onset and length values.");
			return false;
		} else {
			return true;
		}
	}
	
	/** loads onset information for each run and condition from input files 
	 *  (one for each run)
	 *   
	 * @param fileList - format: onsetFileRun1 onsetFileRun2 onsetFileRun3 ...
	 * @param isBlock true if block session file; false if event-related
	 * @return arrayList 
	 * @throws IOException 
	 */
	private static ArrayList<String> loadOnsetsFromFiles(ArrayList<String> fileList, 
			boolean isBlock, int nConds) {
		
		ArrayList<String> onsetValues = new ArrayList<String>();
		for (int r = 0; r < fileList.size(); ++r) {
			String filename = fileList.get(r);
			String currRunOnsets = "";
			// Reads the onset values from the file first to verify
			// that they are all valid. This is to prevent setting
			// invalid values in the table.
			try {
			BufferedReader br = new BufferedReader(new FileReader(filename));
			String line = br.readLine();
			int nCondsInFile = 0;
			while (line != null && line.length() > 0) {
				nCondsInFile++;
				// Checks the line of onset values.
				if (!areOnsetValuesValid(line, "Onset file " + filename)) {
					br.close();
					return null;
				}
				String[] onsetsLine = line.split("\\s");
				for (String o: onsetsLine) {
					currRunOnsets = currRunOnsets.concat(o + ",");
				}
				currRunOnsets = currRunOnsets.substring(0, currRunOnsets.length() - 1);
				currRunOnsets = currRunOnsets.concat(":");
				
//				System.out.println("curr run onsets: " + currRunOnsets);
				
				line = br.readLine();
				
				// Checks the line of length values next if the session file
				// is blocked fMRI.
				if (line != null && isBlock) {
					if (!areLengthValuesValid(line, "Onset file " + filename)) {
						br.close();
						return null;
					}
					String lengthsLine = line;
					
					// Checks if the number of onset values is the same as
					// the number of length values.
					if (!sameLength(line, lengthsLine, "Onset file " + filename)) {
						br.close();
						return null;
					}
					
					//onsetValues.add(currRunOnsets.trim());
					line = br.readLine();
				}
			}
			br.close();
			
			// Checks that the number of values in the file match the
			// number of conditions
			if (nCondsInFile != nConds) {
				System.out.println("Number of rows in onset file " + filename + " does not match" +
						" number of conditions.");
				return null;
			}
			
			currRunOnsets = currRunOnsets.substring(0, currRunOnsets.length() - 1);
			onsetValues.add(currRunOnsets);
		} 
		
		catch (Exception e) {
			System.out.println("Onset file " + filename + " could not be loaded.");
    		return null;
		}
		}
		return onsetValues;
	
	}

	private static void createDatamat(String[] argv) {
		Map<String, ArrayList<String> > params = getParams(argv);
		if (params == null) {
			return;
		}
		
		StreamedProgressHelper helper = new StreamedProgressHelper();
		helper.addStream(System.out);
		
		String sessionFile = "";
		boolean isBlock = false;
		if (params.containsKey("SESSION_FILE") && params.get("SESSION_FILE").size() > 0) {
			sessionFile = params.get("SESSION_FILE").get(0);
			if (sessionFile.endsWith("BfMRIsession.mat")) {
				isBlock = true;
			}
		} else {
			System.out.println("You must specify a session file to use.");
			return;
		}
		
		double coordinateThreshold = 0.15;
		if (params.containsKey("COORD_THRESH") && params.get("COORD_THRESH").size() > 0) {
			coordinateThreshold = Double.parseDouble(params.get("COORD_THRESH").get(0) );
		}
		
		int[] ignoreSlices = null;
		if (params.containsKey("IGNORE_SLICES") && params.get("IGNORE_SLICES").size() > 0) {
			ArrayList<String> ints = params.get("IGNORE_SLICES");
			ignoreSlices = new int[ints.size()];
			for (int i = 0; i < ints.size(); ++i) {
				try {
					ignoreSlices[i] = Integer.parseInt(ints.get(i));
				} catch (NumberFormatException nfex) {
					System.out.println("Slices to ignore must be integers.");
					return;
				}
			}
		}
		
		boolean normalizeMeanVolume = false;
		if (params.containsKey("NORM_MEAN_VOL") && params.get("NORM_MEAN_VOL").size() > 0) {
			normalizeMeanVolume = Boolean.parseBoolean(params.get("NORM_MEAN_VOL").get(0) );
		}
		
		int windowSize = 1; // trivial window size for block data
		if (!isBlock) {
			windowSize = 8; // default for event-related data
		}
		if (params.containsKey("WINDOW_SIZE") && params.get("WINDOW_SIZE").size() > 0) {
			windowSize = Integer.parseInt(params.get("WINDOW_SIZE").get(0) );
		} 
		if (windowSize < 1) {
			System.out.println("Window size must be at least one (1).");
			return;
		}
		if (isBlock && windowSize != 1) {
			System.out.println("Window size must be 1 for Block PLS.");
			return;
		}
		
		SessionProfile profile;
		try {
			// always read in session profile as if it's PLS
			profile = SessionProfile.loadSessionProfile(sessionFile, windowSize==1, false);
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
		
		boolean normalizeWithRefScans = true;
		if (params.containsKey("NORM_REF_SCANS") && params.get("NORM_REF_SCANS").size() > 0) {
			normalizeWithRefScans = Boolean.parseBoolean(params.get("NORM_REF_SCANS").get(0) );
		}
		
		boolean considerAllVoxels = coordinateThreshold == 0.0;
		 
		boolean singleSubject = false;
		if (params.containsKey("SINGLE_SUBJECT") && params.get("SINGLE_SUBJECT").size() > 0) {
			singleSubject = Boolean.parseBoolean(params.get("SINGLE_SUBJECT").get(0) );
		}
		
		RunGenerateDatamat worker = new RunGenerateDatamat(
				profile.isBlock, 
				profile.ignoreRuns,
				sessionFile,
				profile.useBrainMask,
				profile.brainMaskFile,
				coordinateThreshold, 
				ignoreSlices,
				normalizeMeanVolume,
				profile.numSkippedScans,
				windowSize, 
				profile.mergeAcrossRuns,
				normalizeWithRefScans,
				considerAllVoxels, 
				singleSubject,
				profile.conditionInfo,
				profile.runInfo,
				profile.datamatPrefix);
		
		worker.progress = helper;
		worker.start();
	}
	
	private static void setupPlsAnalysis(String[]argv) {
		Map<String, ArrayList<String> > params = getParams(argv);
		
		String plsType = "";
		if (params.containsKey("PLSTYPE") && params.get("PLSTYPE").size() > 0) {
			plsType = params.get("PLSTYPE").get(0).toUpperCase();
		}
		// Only 'MEAN-CENTERING_PLS' is currently available
		else plsType = "MEAN-CENTERING_PLS"; 
		
//		if (!plsType.equals("MEAN-CENTERING_PLS")) {
//			System.out.println("The only PLS type currently available is 'MEAN-CENTERING_PLS'.");
//			return;
//		}
		if (!plsType.equals("MEAN-CENTERING_PLS") && !plsType.equals("BEHAVIOR_PLS") &&
				!plsType.equals("NON-ROTATED_TASK_PLS") && !plsType.equals("MULTIBLOCK_PLS") && !plsType.equals("NON-ROTATED_BEHAVIOR_PLS") ) {
			System.out.println("The pls type must be one of:");
			System.out.println("mean-centering_PLS, behavior_PLS, non-rotated_task_PLS, multiblock_PLS, non-rotated_behavior_pls");
			return;
		}
		
		boolean isBlock = false;
		if (params.containsKey("BLOCK") && params.get("BLOCK").size() > 0) {
			isBlock = Boolean.parseBoolean(params.get("BLOCK").get(0) );
		}
		
		String fileName = "";
		if (params.containsKey("SETUP_FILE_PREFIX") && params.get("SETUP_FILE_PREFIX").size() > 0) {
			fileName = params.get("SETUP_FILE_PREFIX").get(0) + PlsAnalysisSetupFileFilter.EXTENSION;
		} else {
			System.out.println("Must include SETUP_FILE_PREFIX.");
			return;
		}
		
		Vector<String[]> sessionProfiles = new Vector<String[]>();
		if (params.containsKey("SESSION_FILES") && params.get("SESSION_FILES").size() > 0) {
			for(String s : params.get("SESSION_FILES") ) {
				sessionProfiles.add(s.split(","));
			}
		}
		if (sessionProfiles.size() == 0) {
			System.out.println("You must have at least one group with at least one session profile.");
			return;
		}
		
		String contrastFilename = "";
		if (plsType.equals("NON-ROTATED_TASK_PLS") || plsType.equals("NON-ROTATED_BEHAVIOR_PLS")) {
			if (params.containsKey("CONTRAST_FILENAME") && params.get("CONTRAST_FILENAME").size() > 0) {
				contrastFilename = params.get("CONTRAST_FILENAME").get(0);
			}
			else {
				System.out.println("Must include contrast filename when doing non-rotated task PLS.");
				return;
			}
		}
		
		String behaviorFilename = "";
		if (plsType.equals("BEHAVIOR_PLS") || plsType.equals("MULTIBLOCK_PLS") || plsType.equals("NON-ROTATED_BEHAVIOR_PLS")) {
			if (params.containsKey("BEHAVIOR_FILENAME") && params.get("BEHAVIOR_FILENAME").size() > 0) {
				behaviorFilename = params.get("BEHAVIOR_FILENAME").get(0);
			}
			else {
				System.out.println("Must include behavior filename when doing behavior or multiblock PLS.");
				return;
			}
		}
				
		String resultsFilename = "";
		if (params.containsKey("RESULTS_FILE_PREFIX") && params.get("RESULTS_FILE_PREFIX").size() > 0) {
			String extension = fMRIResultFileFilter.EXTENSION;
			if (isBlock) {
				extension = BfMRIResultFileFilter.EXTENSION;
			}
			resultsFilename = params.get("RESULTS_FILE_PREFIX").get(0) + extension;
		}
		else {
			System.out.println("Must include results file prefix.");
			return;
		}
		
		Vector<Integer> conditionSelection = new Vector<Integer>();
		if (params.containsKey("CONDITION_SELECTION") ) {
			for (String s : params.get("CONDITION_SELECTION") ) {
				try {
					conditionSelection.add(Integer.parseInt(s) );
				} catch (NumberFormatException e) {
					System.out.println("Condition selection must be a list of integers.");
				}
			}
		}
		
		Vector<Integer> behaviorBlockConditionSelection = new Vector<Integer>();
		if (plsType.equals("MULTIBLOCK_PLS") && params.containsKey("BEHAVIOR_BLOCK_CONDITION_SELECTION") ) {
			for (String s : params.get("BEHAVIOR_BLOCK_CONDITION_SELECTION") ) {
				try {
					behaviorBlockConditionSelection.add(Integer.parseInt(s) );
				} catch (NumberFormatException e) {
					System.out.println("Behavior block condition selection must be a list of integers.");
				}
			}
		}
		
		int numPermutations = 0;
		if (params.containsKey("NUM_PERMUTATIONS") && params.get("NUM_PERMUTATIONS").size() > 0) {
			try {
				numPermutations = Integer.parseInt(params.get("NUM_PERMUTATIONS").get(0) );
				if (numPermutations < 0) {
					System.out.println("Number of permutations cannot be negative.");
					return;
				}
			} catch (NumberFormatException e) {
				System.out.println("Number of permutations must be an integer.");
				return;
			}
		}
		
		int numBootstraps = 0;
		if (params.containsKey("NUM_BOOTSTRAPS") && params.get("NUM_BOOTSTRAPS").size() > 0) {
			// make sure we have at least 3 session files/group
			for (String[] sf : sessionProfiles) {
				if (sf.length < 3) {
					System.out.println("Must have at least 3 session files / group to perform bootstrap.");
					return;
				}
			}
			try {
				numBootstraps = Integer.parseInt(params.get("NUM_BOOTSTRAPS").get(0) );
				if (numBootstraps < 0) {
					System.out.println("Number of bootstraps cannot be negative.");
					return;
				}
			} catch (NumberFormatException e) {
				System.out.println("Number of bootstraps must be an integer.");
				return;
			}
		}
		
		double confidenceLevel = 95;
		if (params.containsKey("CONFIDENCE_LEVEL") && params.get("CONFIDENCE_LEVEL").size() > 0) {
			try {
				confidenceLevel = Double.parseDouble(params.get("CONFIDENCE_LEVEL").get(0) );
				if (confidenceLevel < 0 || confidenceLevel > 100) {
					throw new NumberFormatException();
				}
			} catch (NumberFormatException e) {
				System.out.println("Confidence level must be a number between 0 and 100.");
				return;
			}
		}
		
		// Save the pls analysis setup file
		int maxNumSessions = 0;
		int numGroups = sessionProfiles.size();
		if (!(numGroups > 0)) {
			JOptionPane.showMessageDialog(null, "Error saving PLS setup file " + fileName + " - " + 
					"must include at least one session file.",
					"Error", JOptionPane.ERROR_MESSAGE);
			return;
		}
		for (int i = 0; i < numGroups; i++) {
			int currNumSessions = sessionProfiles.get(i).length;
			maxNumSessions = Math.max(maxNumSessions, currNumSessions);
		}
		
		MLStructure plsSetupInfo = new MLStructure("pls_setup_info", new int[] {1, 1});
		
		// Save the session file info for each group.
		MLStructure sessionFileInfo = new MLStructure("session_file_info", new int[] {1, numGroups});
		for (int i = 0; i < numGroups; i++) {
			int currNumSessions = sessionProfiles.get(i).length;
			MLCell currSessionFiles = new MLCell("session_files" + i, new int[] {1, currNumSessions});
			for (int sf = 0; sf < currNumSessions; sf++) {
				currSessionFiles.set(new MLChar("session_file" + sf, sessionProfiles.get(i)[sf]), 0, sf);
			}
			sessionFileInfo.setField("session_files", currSessionFiles, i);
			sessionFileInfo.setField("grp_size", new MLDouble("grp_size" + i, new double[][]{{currNumSessions}}), i);
		}
		plsSetupInfo.setField("session_file_info", sessionFileInfo);
		
		// Save condition selection info.

		String sessProfFileName = sessionProfiles.get(0)[0];
		Vector<Integer> condSelectTmp = null;
		try {
			condSelectTmp = selectAllConds(sessProfFileName);
		}
		catch (FileNotFoundException e) {
			System.out.println("Could not find PLS session file " + sessProfFileName +  ".");
			return;

		} catch (IOException e) {
			System.out.println("Could not load PLS session file " + sessProfFileName + ".");
			return;
		}

		if (conditionSelection.size() == 0) {
			conditionSelection = condSelectTmp;
		}

		int numCondsInSessFile = condSelectTmp.size();
		int numConditions = conditionSelection.size();
		if (numCondsInSessFile != numConditions) {
			System.out.println("Wrong number of elements in condition selection input. " +
					"\nExpected:  " + numCondsInSessFile + " elements.");
			return;
		}
		
		MLDouble condSelect = new MLDouble("cond_selection", new int[]{numConditions, 1});
		for (int i = 0; i < numConditions; ++i) {
			condSelect.set(new Double(conditionSelection.get(i)), i);
		}
		plsSetupInfo.setField("cond_selection", condSelect);
		
		// Save behavior block condition selection info for multiblock PLS
		numConditions = behaviorBlockConditionSelection.size();
		if (plsType.equals("MULTIBLOCK_PLS")) {
			Vector<Integer> behavBlockCondSelectTmp = null;
			try {
				behavBlockCondSelectTmp = selectAllConds(sessProfFileName);
			} 
			catch (FileNotFoundException e) {
				System.out.println("Could not find PLS session file " + sessProfFileName + ".");
				return;
			} 
			catch (IOException e) {
				System.out.println("Could not load PLS session file " + sessProfFileName + ".");
				return;
			}

			if (behaviorBlockConditionSelection.size() == 0) {
				behaviorBlockConditionSelection = behavBlockCondSelectTmp;
				numConditions = behaviorBlockConditionSelection.size(); // update from size 0
			}

			numCondsInSessFile = behavBlockCondSelectTmp.size();
			if (numCondsInSessFile != numConditions) {
				System.out.println("Wrong number of elements in behavior block condition selection input. " +
						"\nExpected:  " + numCondsInSessFile + " elements.");
				return;
			}
		}
//		System.out.println("Num conditions: " + numConditions);
//		System.out.println("Behav block cond selection: ");
//		System.out.println(behaviorBlockConditionSelection.toString());
		condSelect = new MLDouble("behav_block_cond_selection", new int[]{numConditions, 1});
		for (int i = 0; i < numConditions; ++i) {
			condSelect.set(new Double(behaviorBlockConditionSelection.get(i)), i);
		}
		plsSetupInfo.setField("behav_block_cond_selection", condSelect);
		
		
		// Save PLS analysis type info.
		
		if (plsType.equals("MEAN-CENTERING_PLS")) {
			plsSetupInfo.setField("mean-centering_PLS", new MLDouble("mean-centering_PLS", new double[][]{{1}}));
		} else {
			plsSetupInfo.setField("mean-centering_PLS", new MLDouble("mean-centering_PLS", new double[][]{{0}}));
		}
		if (plsType.equals("BEHAVIOR_PLS")) {
			plsSetupInfo.setField("behavior_PLS", new MLDouble("behavior_PLS", new double[][]{{1}}));
		} else {
			plsSetupInfo.setField("behavior_PLS", new MLDouble("behavior_PLS", new double[][]{{0}}));
		}
		if (plsType.equals("NON-ROTATED_TASK_PLS")) {
			plsSetupInfo.setField("non-rotated_task_PLS", new MLDouble("non-rotated_task_PLS", new double[][]{{1}}));
		} else {
			plsSetupInfo.setField("non-rotated_task_PLS", new MLDouble("non-rotated_task_PLS", new double[][]{{0}}));
		}
//		if (plsType.equals("MULTIBLOCK_PLS")) {
//			plsSetupInfo.setField("multiblock_PLS", new MLDouble("multiblock_PLS", new double[][]{{1}}));
//		} else {
//			plsSetupInfo.setField("multiblock_PLS", new MLDouble("multiblock_PLS", new double[][]{{0}}));
//		}
		if (plsType.equals("NON-ROTATED_BEHAVIOR_PLS")) {
			plsSetupInfo.setField("non-rotated_behavior_PLS", new MLDouble("non-rotated_behavior_PLS", new double[][]{{1}}));
		} else {
			plsSetupInfo.setField("non-rotated_behavior_PLS", new MLDouble("non-rotated_behavior_PLS", new double[][]{{0}}));
		}
		
		// Save data file fields.
		plsSetupInfo.setField("contrast_data_filename", new MLChar("contrast_data_filename", contrastFilename));
		plsSetupInfo.setField("behavior_data_filename", new MLChar("behavior_data_filename", behaviorFilename));
				
		// Save permutations/bootstrap info.
		String numPerms = Integer.toString(numPermutations);
		plsSetupInfo.setField("num_permutations", new MLChar("num_permutations", numPerms));
		
		String numBoots = Integer.toString(numBootstraps);
		plsSetupInfo.setField("num_bootstraps", new MLChar("num_bootstraps", numBoots));
		
		String confidenceLevelString = Double.toString(confidenceLevel);
		plsSetupInfo.setField("confidence_level", new MLChar("confidence_level", confidenceLevelString));
		
		// Save the results file name.
		plsSetupInfo.setField("results_filename", new MLChar("results_filename", resultsFilename));
	
		ArrayList<MLArray> list = new ArrayList<MLArray>();
		list.add(plsSetupInfo);
		try {
			new MatFileWriter(fileName, list);
		} catch (Exception e) {
			System.out.println("Could not save to PLS setup file " + fileName + ".");
			return;
		}
	}

	private static Vector<Integer> selectAllConds(String sessProfFileName) throws FileNotFoundException,
	 	IOException {
		Vector<Integer> condSelectTmp = null;
		MLStructure sessProfStruct = (MLStructure) new NewMatFileReader(sessProfFileName).getContent().get("session_info");
		String[] conditions = MLFuncs.MLCell1dRow2StrArray((MLCell)sessProfStruct.getField("condition"));
		condSelectTmp = new Vector<Integer>(conditions.length);
		for (int i = 0; i < conditions.length; i++) {
			condSelectTmp.add(new Integer(1));
		}

		return condSelectTmp;
	}
	
	
	private enum SplitType { SESSION, RUN };
	private static void setupNpairsAnalysis(String[] argv) {
		Map<String, ArrayList<String>> params = getParams(argv);
		MLStructure npairsSetupInfo = new MLStructure("npairs_setup_info", new int[] {1, 1});	
		
		// If 'Event-related' analysis, datamats are loaded; if 'Blocked', data is read in
		// from image files
		boolean isBlocked = true;
		int loadDatamats = 0;
		if (params.containsKey("BLOCK")) {
			isBlocked = Boolean.parseBoolean(params.get("BLOCK").get(0) );
			if (!isBlocked) {
				loadDatamats = 1;
			}
		}
		npairsSetupInfo.setField("load_datamats", new MLDouble("loadDatamats", 
				new double[][]{{loadDatamats}}));
		String splitType = "Session (default)";
		if (params.containsKey("SPLIT_TYPE")) {
			if (params.get("SPLIT_TYPE").size() > 0) {
				String s = params.get("SPLIT_TYPE").get(0).toUpperCase();
				SplitType enteredSplitType = SplitType.valueOf(s);
			
				switch(enteredSplitType) {
				
					case SESSION: splitType = "Session (default)"; 
					 	break;
					case RUN: splitType = "Run";
						break;
					default: 
						System.out.println("Split type must be either SESSION or RUN.");
						return;
				}
			}
			if (splitType.equals("Run") && !isBlocked) {
				System.out.println("Cannot split data by run when doing event-related NPAIRS analysis.");
				return;
			}
		}
		
		Vector<String[]> sessionProfiles = new Vector<String[]>();
		if (params.containsKey("SESSION_FILES") ) {
			for (String s : params.get("SESSION_FILES") ) {
				String[] group = s.split(",");
				if (splitType.equals("Session (default)") && group.length < 2) {
					System.out.println("You must include at least two session files in each group.");
					return;
				}
				sessionProfiles.add(group);
			}
		}
		int numGroups = sessionProfiles.size();
		if (numGroups <= 0) {
			System.out.println("You must include at least one group of session files.");
			return;
		}
		
		
		Vector<Integer> classSelection = new Vector<Integer>();
		if (params.containsKey("CLASS_SELECTION") ) {
			String classSelErrorMessage = "Class selection elements must be either 1 (to include corresponding class) or " +
							"\n0 (to exclude corresponding class).  Class order is taken from first session file " +
							"in first group.";
			for (String s : params.get("CLASS_SELECTION") ) {
			
					Integer i = Integer.parseInt(s);
					if (i != 0 && i != 1) {
						System.out.println(classSelErrorMessage);
						return;
					}
					classSelection.add(i);
			}
		}
		
		String sessProfFileName = sessionProfiles.get(0)[0];
		Vector<Integer> classSelectTmp = null;
		try {
			classSelectTmp = selectAllConds(sessProfFileName);
		}
		catch (FileNotFoundException e) {
			System.out.println("Could not find NPAIRS session file " + sessProfFileName +  ".");
			return;

		} catch (IOException e) {
			System.out.println("Could not load NPAIRS session file " + sessProfFileName + ".");
			return;
		}

		if (classSelection.size() == 0) {
			classSelection = classSelectTmp;
		}

		int numCondsInSessFile = classSelectTmp.size();
		int numConditions = classSelection.size();
		if (numCondsInSessFile != numConditions) {
			System.out.println("Wrong number of elements in class selection input. " +
					"\nExpected:  " + numCondsInSessFile + " elements.");
			return;
		}
		
		int maxNumSessions = 0;
		for (int i = 0; i < numGroups; ++i) {
			int currNumSessions = sessionProfiles.get(i).length;
			maxNumSessions = Math.max(maxNumSessions, currNumSessions);
		}

		ArrayList<String> ignoreRuns = null;
		// get ignore runs info from setup file options if included 
		if (params.containsKey("IGNORE_RUNS") ) {
			ignoreRuns = params.get("IGNORE_RUNS");
		}

		// Save sessionfile info for each group
		MLStructure sessionFileInfo = new MLStructure("session_file_info", new int[] {1, numGroups});
		// TODO: refine test for valid split object partition info; should consider how splits
		// include proportion of data from each grp when analyzing multiple groups 
		// (Currently just check total number of input session files to determine valid split
		// partition entries.)
		int nSessFiles = 0;
		int nRunsInCurrGrp = 0;
		int nRunsTotal = 0;
		for (int g = 0; g < numGroups; ++g) {
			int currNumSessions = sessionProfiles.get(g).length;
			nSessFiles += currNumSessions;
			MLCell currSessionFiles = new MLCell("session_files" + g, new int[] {1, currNumSessions});
			MLCell currIgnoreRuns = new MLCell("ignore_runs" + g, new int[] {1, currNumSessions});
			
			boolean getIgnoreRunsFromSessFile = false;
			ArrayList<String> currGrpIgnRuns = null;
			int[] numIgnoredRuns = null;
			if (ignoreRuns != null) {
				System.out.println("Getting ignore runs info from setup options...");
				// get ignore runs info from setup options if included
				String[] sCurrGrpIgnRuns = ignoreRuns.get(g).split(":");
				int nSF = sCurrGrpIgnRuns.length;
				if (nSF != currNumSessions) {
					System.out.println("Must provide IGNORE_RUNS information for all session files if including this option.");
					return;
				}
				
				currGrpIgnRuns = new ArrayList<String>(nSF);
				numIgnoredRuns = new int[nSF];
				int c = 0;
				for (String sCurrSFIgnRuns : sCurrGrpIgnRuns) {
					String[] individuals = sCurrSFIgnRuns.split(",");
					String sIgnRuns = "";
					for (String s : individuals) {
						if (Integer.parseInt(s) != 0) {
							sIgnRuns += s + " ";
							numIgnoredRuns[c] += 1;
						}
					}
					++c;
					sIgnRuns = sIgnRuns.trim();
					currGrpIgnRuns.add(sIgnRuns);
					
				}
			}
			else {
				getIgnoreRunsFromSessFile = true;
				System.out.println("Getting ignore runs info from session files...");
			}
			
			for(int sf = 0; sf < currNumSessions; ++sf) {
				String currSessFile = sessionProfiles.get(g)[sf];
				currSessionFiles.set(new MLChar("session_file" + sf, currSessFile), 0, sf);
				if (!getIgnoreRunsFromSessFile) {
					currIgnoreRuns.set(new MLChar("ignore_run" + sf, currGrpIgnRuns.get(sf)), 0, sf);
				}
				try {
					MLStructure currSessFileInfo = (MLStructure)new NewMatFileReader(currSessFile).
						getContent().get("session_info");
					int currNumRuns = ((MLDouble)currSessFileInfo.getField("num_runs")).getIntArray()[0][0];
					// remove ignored runs from count
					
					if (getIgnoreRunsFromSessFile) {
						
//						////////////DEBUG/////////////////////
//						System.out.println("Getting ignore runs info from session file # " + sf + ", grp #" + g + ".");
						
						String ignRunsStr = ((MLChar)currSessFileInfo.getField("runs_skipped")).getString(0);
						String[] ignoredRuns = ignRunsStr.split("\\s");
						int numIgnored = 0;
						for (String i : ignoredRuns) {
							if (new Integer(i) > 0) {
								++numIgnored;			
							}
						}
						currNumRuns -= numIgnored;
						currIgnoreRuns.set(new MLChar("ignore_run" + sf, ignRunsStr), 0, sf);
						
					}
					else { 
						// ignore runs info is in setup options
						currNumRuns -= numIgnoredRuns[sf];
					}
					nRunsInCurrGrp += currNumRuns;
					
				} catch (FileNotFoundException e) {
					System.out.println("Could not find NPAIRS session file " + currSessFile +  ".");
					return;

				} catch (IOException e) {
					System.out.println("Could not load NPAIRS session file " + currSessFile + ".");
					return;
				}
			}
			
			sessionFileInfo.setField("session_files", currSessionFiles, g);
			sessionFileInfo.setField("grp_size", new MLDouble("grp_size" + g, 
						new double[][]{{currNumSessions}}), g);
			sessionFileInfo.setField("ignore_runs", currIgnoreRuns, g);
			
			// If Run is split type, must have at least 2 runs to split on.  If Session is
			// split type, will already have checked for at least 2 session files, which 
			// automatically implies at least 2 runs. 
			if (nRunsInCurrGrp < 2) {
				System.out.println("You must include at least 2 runs in each group.");
				return;
			}
			nRunsTotal += nRunsInCurrGrp;
			nRunsInCurrGrp = 0;		
		}
		
		
		npairsSetupInfo.setField("split_type", new MLChar("split_type", splitType));
		npairsSetupInfo.setField("session_file_info", sessionFileInfo);
		
		int numClasses = classSelection.size();
		MLDouble classSelect = new MLDouble("class_selection", new int[]{numClasses, 1});
		for (int i = 0; i < numClasses; ++i) {
			classSelect.set(new Double(classSelection.get(i)), i);
		}
		npairsSetupInfo.setField("class_selection", classSelect);
		
		// Save NPAIRS analysis type info
		int doMSR = 1;
		if (params.containsKey("DO_MSR") && !Boolean.parseBoolean(params.get("DO_MSR").get(0)) ) {
				doMSR = 0;
		}
		npairsSetupInfo.setField("do_msr", new MLDouble("do_msr", new double[][] {{doMSR}}));
		
		int doGLM = 0;
		if (params.containsKey("DO_GLM") && Boolean.parseBoolean(params.get("DO_GLM").get(0)) ) {
			//doGLM = 1;
			System.out.println("NPAIRS Analysis Modelling Options: GLM is not implemented.  " +
					"\nPCA + CVA is the only NPAIRS analysis option currently implemented.");
			return;
		}
		npairsSetupInfo.setField("do_glm", new MLDouble("do_glm", new double[][]{{doGLM}}));
		
		// PCA + CVA are set to true by default since that's the only option right now
		int doPCA = 1;
		if (params.containsKey("DO_PCA") && !Boolean.parseBoolean(params.get("DO_PCA").get(0)) ) {
			doPCA = 0;
		}	
		npairsSetupInfo.setField("do_pca", new MLDouble("do_pca", new double[][]{{doPCA}}));
		
		int doCVA = 1;
		if (params.containsKey("DO_CVA") && !Boolean.parseBoolean(params.get("DO_CVA").get(0)) ) {
				doCVA = 0;
		} 
		npairsSetupInfo.setField("do_cva", new MLDouble("do_cva", new double[][]{{doCVA}}));
		
		if (params.containsKey("CVA_CLASSFILE") && params.get("CVA_CLASSFILE").size() > 0) {
					String cvaClassFile = params.get("CVA_CLASSFILE").get(0);
					npairsSetupInfo.setField("cva_class_file", new MLChar("cva_class_file", cvaClassFile));
		}
		
		if (doCVA == 1 && doPCA == 1) {
			int normPCs = 0;
			if (params.containsKey("NORM_PCS") ) {
				if (params.get("NORM_PCS").size() == 0 || params.get("NORM_PCS").get(0).equals("true") ) {
					normPCs = 1;
				}
			}
			npairsSetupInfo.setField("norm_pcs", new MLDouble("norm_pcs", 
					new double[][]{{normPCs}}));
		}
		else {
			// for now, must do PCA + CVA
			System.out.println("NPAIRS Analysis Modelling Options: PCA + CVA is the only NPAIRS analysis " +
					"option currently implemented.");
			return;
		}

		boolean doResampling = true;
		int splitHalfXvalid = 1; // it's the only kind implemented right now
		int bootstrap = 0;
		if (params.containsKey("DO_RESAMP") && params.get("DO_RESAMP").size() > 0) {
			if (params.get("DO_RESAMP").get(0).equals("false") ) {
				doResampling = false;
				splitHalfXvalid = 0; 
			}
		}
		
		if (doResampling) { 
			// eventually will need RESAMP_TYPE but right now split_half_xvalid is 
			// automatically set when DO_RESAMP = true
			if (params.containsKey("RESAMP_TYPE") && params.get("RESAMP_TYPE").size() > 0) {
				if (params.get("RESAMP_TYPE").get(0).equals("split_half_xvalid") ) {
					splitHalfXvalid = 1;
				} else if (params.get("RESAMP_TYPE").get(0).equals("bootstrap") ) {
					System.out.println("Bootstrap not implemented.");
					return;
					//splitHalfXvalid = 0;
					//bootstrap = 1;
				} else {
					System.out.println("Invalid RESAMP_TYPE.");
					return;
				}
			}
		
			npairsSetupInfo.setField("split_half_xvalid", new MLDouble(
					"split_half_xvalid", new double[][]{{splitHalfXvalid}}));
			
			npairsSetupInfo.setField("bootstrap", new MLDouble("bootstrap",
					new double[][]{{bootstrap}}));
		}

		int numSplits = 500;
		if (doResampling) {
			if (params.containsKey("NUM_SPLITS") && params.get("NUM_SPLITS").size() > 0) {
				numSplits = Integer.parseInt(params.get("NUM_SPLITS").get(0));
				
			}
			npairsSetupInfo.setField("num_splits", new MLDouble("num_splits", 
					new double[][]{{numSplits}}));
		}

		if (doResampling) {
			int nSplitObj = nSessFiles;
			if (splitType.equals("Run")) {
				nSplitObj = nRunsTotal;
			}
			if (params.containsKey("SPLITS_INFO_FILENAME") && params.get("SPLITS_INFO_FILENAME").size() > 0) {
				String splitsInfoFilename = params.get("SPLITS_INFO_FILENAME").get(0);
				npairsSetupInfo.setField("splits_info_filename", new MLChar("splits_info_filename",
						splitsInfoFilename));
			} else if (params.containsKey("SPLIT_PARTITION") && params.get("SPLIT_PARTITION").size() > 1) {
				String splitPartStr[] = new String[2];
				splitPartStr[0] = params.get("SPLIT_PARTITION").get(0);
				splitPartStr[1] = params.get("SPLIT_PARTITION").get(1);
				
				String splitPartErrorMessage = "Invalid split partition values.\n" +
						"Values must be integers indicating number of split objects " +
						"in each split half.";
				try {
					int[] splitPartition = new int[2];
					splitPartition[0] = Integer.parseInt(splitPartStr[0]);
					splitPartition[1] = Integer.parseInt(splitPartStr[1]);
					if (splitPartition[0] <= 0 || splitPartition[1] <= 0) {
						System.out.println(splitPartErrorMessage);
						return;
					}
					if (splitPartition[0] + splitPartition[1] > nSplitObj) {
						System.out.println(splitPartErrorMessage);
						return;
					}
					npairsSetupInfo.setField("split_partition", new MLDouble("split_partition",
							new double[][]{{splitPartition[0], splitPartition[1]}}));
				}
				catch (NumberFormatException nfe) {
					System.out.println(splitPartErrorMessage);
					return;
				}
			}
			else { // set split partition explicitly
				int[] splitPartition = new int[2];
				splitPartition[0] = nSplitObj / 2;
				splitPartition[1] = nSplitObj - splitPartition[0];
				npairsSetupInfo.setField("split_partition", new MLDouble("split_partition",
						new double[][]{{splitPartition[0], splitPartition[1]}}));
			}
		}
		
		// Save initial feature selection info 
		boolean doFeatureSelection = true;
		if (params.containsKey("DO_INIT_EVD") && params.get("DO_INIT_EVD").get(0).equals("false") ) {
				doFeatureSelection = false;
		}
		
		if (doFeatureSelection) {
			npairsSetupInfo.setField("do_init_svd", new MLDouble("do_init_svd", 
					new double[][]{{1}}));
		} else {
			npairsSetupInfo.setField("do_init_svd", new MLDouble("do_init_svd", 
					new double[][]{{0}}));
		}
		
		if (doFeatureSelection) {
			int loadEVD = 0; // default false
			//			if (params.containsKey("LOAD_EVD") && (params.get("LOAD_EVD").size() <= 0 
			//					|| params.get("LOAD_EVD").get(0).equals("true")) ) {
			//				loadEVD = 1;

			String svdFilePrefix = "";
			if (params.containsKey("EVD_FILE_PREFIX")) {
				if (params.get("EVD_FILE_PREFIX").size() > 0) {
					loadEVD = 1;
					svdFilePrefix = params.get("EVD_FILE_PREFIX").get(0);
				}
				else {
					System.out.println("Please enter EVD file prefix to load EVD information.");
					return;
				}
			}

			npairsSetupInfo.setField("svd_file_prefix", new MLChar("svd_file_prefix", 
					svdFilePrefix));

			npairsSetupInfo.setField("load_svd", new MLDouble("load_svd",
					new double[][]{{loadEVD}}));

			double dataReductFactor = 0.3;
			if (params.containsKey("DRF") && params.get("DRF").size() > 0) {
				dataReductFactor = Double.parseDouble(params.get("DRF").get(0));
				if (dataReductFactor > 1 || dataReductFactor <= 0) {
					System.out.println("DRF must be > 0.0 and <= 1.0.");
					return;
				}
			}
			npairsSetupInfo.setField("drf", new MLDouble("drf", new double[][]{{dataReductFactor}}));
		}
		
		if (doPCA == 1 && doCVA == 1) {
			if (params.containsKey("SET_PC_RANGE") ) {
				if (!doResampling) {
					System.out.println("Cannot set pc range if not doing resampling.");
					return;
				}
				if (params.containsKey("PCS_SPLIT") || (params.containsKey("PCS_ALL_DATA"))) {
					System.out.println("Cannot set PC range and specify -PCS_SPLIT or -PCS_ALL_DATA at the same time.");
					return;
				}
				if (params.get("SET_PC_RANGE").size() >= 3) {
					String pcRange = params.get("SET_PC_RANGE").get(0);
					double pcStep = Double.parseDouble(params.get("SET_PC_RANGE").get(1) );
					double pcMultFact = Double.parseDouble(params.get("SET_PC_RANGE").get(2) );

					npairsSetupInfo.setField("pc_range", new MLChar("pc_range", pcRange));
					npairsSetupInfo.setField("pc_step", new MLDouble("pc_step", 
							new double[][]{{pcStep}}));
					npairsSetupInfo.setField("pc_mult_factor", new MLDouble("pc_step", 
							new double[][]{{pcMultFact}}));
				} else {
					System.out.println("Must enter pc range info as: X-Y,S,M where X,Y are range for split-data no. of PCs," +
					"\nS is step size and M is multiplication factor to determine corresponding full-data no. of PCs.");
					return;
				}
			} else if (params.containsKey("PCS_ALL_DATA") && params.get("PCS_ALL_DATA").size() > 0) { 
				// set pcs explicitly
				String pcsForFullData = "";
				for(String s : params.get("PCS_ALL_DATA") ) {
						pcsForFullData += s;
					}
					pcsForFullData = pcsForFullData.trim();
			
//				else {
//					System.out.println("Must include PCs for full data analysis.");
//					return;
//				}
				
				// ONLY READ PCS_SPLIT IF PCS_ALL_DATA IS ALSO SET
				String pcsForSplit = "";
				if (params.containsKey("PCS_SPLIT") && params.get("PCS_SPLIT").size() > 0) {
					for(String s : params.get("PCS_SPLIT") ) {
						pcsForSplit += s;
					}
					pcsForSplit = pcsForSplit.trim();
				} else if (doResampling) {
					System.out.println("Must include PCs for split data analysis or set PC range.");
					return;
				}
				
				npairsSetupInfo.setField("pcs_training", new MLChar("pcs_training",
						pcsForSplit));
				npairsSetupInfo.setField("pcs_all_data", new MLChar("pcs_all_data",
						pcsForFullData));
			} else {
				// no PC information included; set automatically to max possible PC range, step size 
				// NpairsjSetupParams.AUTO_PC_STEP_SIZE
				npairsSetupInfo.setField("set_auto_pc_range", new MLDouble("set_auto_pc_range", 
							new double[][]{{1}}));
				npairsSetupInfo.setField("pc_step", new MLDouble("pc_step", 
							new double[][]{{NpairsjSetupParams.AUTO_PC_STEP_SIZE}}));
			}
			
		}

		// save lots of results files?
		double saveMulti = 0;
		if (params.containsKey("SAVE_LOTS") ) {
			if (params.get("SAVE_LOTS").size() > 0 && params.get("SAVE_LOTS").get(0).equals("true") ) {
				saveMulti = 1;
			}
		}
		npairsSetupInfo.setField("save_multi_files", new MLDouble("save_multi_files",
				new double[][] {{saveMulti}}));
		
		double saveSplits = 0;
		if (params.containsKey("SAVE_SPLITS")) {
			if (saveMulti == 1) {
				if (params.get("SAVE_SPLITS").size() <= 0 || params.get("SAVE_SPLITS").get(0).equals("true") ) {
					saveSplits = 1;
				}
			}
			else if (params.get("SAVE_SPLITS").get(0).equals("true")) {
				System.out.println("Cannot save splits unless -SAVE_LOTS is set to true.");
				return;
			}
			
		}
		npairsSetupInfo.setField("save_split_results", new MLDouble("save_split_results",
				new double[][] {{saveSplits}}));

		String resultsFilename = "npairs";
		if (params.containsKey("RESULTS_FILE_PREFIX") && params.get("RESULTS_FILE_PREFIX").size() > 0) {
			resultsFilename = params.get("RESULTS_FILE_PREFIX").get(0) + "_NPAIRSJresult.mat";
		} else {
			System.out.println("Must include results file prefix.");
			return;
		}
		
		npairsSetupInfo.setField("results_filename", new MLChar("results_filename", 
				resultsFilename));
		
		String analysisFilename = "";
		String analysisSetupSuffix = "_NPAIRSAnalysisSetup.mat";
		if (!isBlocked) analysisSetupSuffix = "_NPAIRSAnalysisSetupER.mat";
		if (params.containsKey("SETUP_FILE_PREFIX") && params.get("SETUP_FILE_PREFIX").size() > 0) {
			analysisFilename = params.get("SETUP_FILE_PREFIX").get(0) + analysisSetupSuffix;
		} else {
			// use results file prefix
			analysisFilename = params.get("RESULTS_FILE_PREFIX").get(0) + analysisSetupSuffix;
		}

		ArrayList<MLArray> list = new ArrayList<MLArray>();
		list.add(npairsSetupInfo);
	
		try {
			File analysisFileDir = new File(analysisFilename).getParentFile();
			if (!analysisFileDir.exists()) {
				String createDirMess = "Creating directory " + analysisFileDir.toString();
				Npairsj.output.println(createDirMess);
				boolean created = analysisFileDir.mkdir();
				if (!created) {
					String errorMessage = "Unable to create directory " + analysisFileDir + ".";
					Npairsj.output.println(errorMessage);
					return;
				}
			}
			System.out.println("Saving npairs setup file " + analysisFilename + "...");
			new MatFileWriter(analysisFilename, list);

		}
		catch(Exception ex) {
			ex.printStackTrace();
			System.out.println("Could not save npairs setup file " + analysisFilename + ".");
			return;
		}
	}
	
	private static void runAnalysis(String[] argv, boolean isNpairs) {
		
		// Check if last arg is 'EVD_ONLY' or an int (indicating analysis no.)
		// instead of a setup param filename
		int lastIdx = argv.length - 1;
		boolean evdOnly = false;
		int analysisNum = -1;
		if (argv[lastIdx].equals("EVD_ONLY")) {
			evdOnly = true;
			--lastIdx; 
		}
		else {
			try {
				analysisNum = Integer.parseInt(argv[lastIdx]);
				--lastIdx;
				
			}
			catch (NumberFormatException nfe) {
			}
		}
		
		// Perform an analysis (or EVD_ONLY) for each setup file given as argument
		for(int n = 2; n <= lastIdx; ++n) {
			String filename = argv[n];
			
			StreamedProgressHelper helper = new StreamedProgressHelper();
			helper.addStream(System.out);
			if (!isNpairs) {

				try {
					Analysis worker = new Analysis(filename);
					worker.progress = helper;
					worker.run();
				} catch (Exception ioex) {
					ioex.printStackTrace();
				}
			} 
			else {
				
				String matlibType = pls.shared.GlobalVariablesFunctions.matrixLibrary;
				String matlibTypeForInitFeatSel = pls.shared.GlobalVariablesFunctions.matrixLibrary;
				boolean loadDatamats = false; 
				if (filename.endsWith("SetupER.mat")) { // 'event-related' NPAIRS
					loadDatamats = true;
				}
				
				// exceptions are caught in ProgressDialogWatcher.run() since they're all generated 
				// in NpairsAnalysis.doTask(); for pls Analysis, though, exceptions are generated outside
				// of Analysis.doTask() hence not caught by ProgressDialogWatcher.run().
				NpairsAnalysis worker = new NpairsAnalysis(filename, matlibType, 
						matlibTypeForInitFeatSel, !loadDatamats, evdOnly, analysisNum);	
				worker.progress = helper;
				worker.run();
			}
		}
	}
	
	private static Map<String, ArrayList<String> > getParams(String[] argv) {
		HashMap<String, ArrayList<String> > params = new HashMap<String, ArrayList<String> >();
		
		String paramName = "other";
		params.put(paramName, new ArrayList<String>() );
		for (int i = 2; i < argv.length; ++i) {
			String param = argv[i];
			
			// Letter check is done because "-1" is a valid onset value 
			// that could be included in "-ONSETS" argument
			if (param.startsWith("-") && Character.isLetter(param.charAt(1))) {
				paramName = param.substring(1);
				params.put(paramName, new ArrayList<String>() );
			} else {
				params.get(paramName).add(param);
			}
		}
		
		return params;
	}
	
	private static void printSessionProfileHelpMessage() {
		System.out.println("Create Session Profile Usage:");
		System.out.println("\nRequired inputs:");
		System.out.println("-SESSION_FILE_PREFIX text -- The file prefix for the session file to be created.");
		System.out.println("-BRAIN_MASK_FILE text -- The name of the mask file to use if using a brain mask." +
				"\n(Required for Block fMRI NPAIRS; optional in all other cases.)");
		System.out.println("-CONDITION_INFO cond1Name cond2Name cond3Name ... -- Name of each condition. ");
		System.out.println("-DATA_FILES run1file1,run1file2 run2file1,run2file2 ... -- Lists of image files to include. These files " +
				"\nwill be relative to the DATA_PATHS parameter. Separate files within runs with commas; separate runs " +
				"\nwith spaces.");
		System.out.println("-DATA_PATHS path_run1 path_run2 path_run3 ... -- List of data paths - 1 for each run.");
		
		System.out.println("\nEither (a) or (b) is required:");
		System.out.println("\n(a)");
		System.out.println("-ONSETS 5,24,43,62,81:14,33,53,72... -- Lists of condition onsets " +
				"(where 'onset' = 1st scan " +
				"\nin condition epoch). Separate onsets with commas (,) and conditions with colons (:). Separate " +
				"\nruns with spaces. Scan numbers are 0-relative.");
		System.out.println("-BLOCK_LENGTHS 7,6,7,7,7:5,7,7,6... -- Lists of lengths (in no. scans) of each " +
				"\ncondition block. Separate blocks for each condition with commas (,) " +
			    "and conditions with colons (:). " +
			    "\nSeparate runs with spaces. (Only used in Block analyses.)");
		System.out.println("\nOR\n");
		System.out.println("(b)");
		System.out.println("-ONSET_FILES onsetsRun1 onsetsRun2 ... -- List of textfiles containing condition onset and (for" +
				"\nBlock fMRI) length information. See user guide for file syntax details. (Optional - default is to " +
				"\nenter onsets using -ONSETS flag and lengths using " +
				"-BLOCK_LENGTHS flag.)");
		System.out.println("\nOptional inputs: ");
		System.out.println("-SESSION_FILE_DIR text -- The directory where the session file is to be saved " +
				"\n(optional; default is to save in path indicated in SESSION_FILE_PREFIX or current working directory " +
				"\nif no path given).");
		System.out.println("-DESCRIPTION text -- An optional description of the session file.");
//		System.out.println("-DATAMAT_PREFIX text -- File prefix (no path) for the datamat file to be created from this session file." +
//				"\n(Default (Block fMRI NPAIRS): datamat prefix == session file prefix; required input in all other cases.)");
		System.out.println("-BLOCK true/false -- True if Blocked fMRI; false if Event-related (default true). ");
		System.out.println("-USE_MASK_FILE true/false -- Optional. This is set to false by default. " +
				"If -BRAIN_MASK_FILE is set, \nthen -USE_MASK_FILE is set to true automatically.  This flag " +
				"would generally only be used when " +
				"\nloading existing session file and changing -USE_MASK_FILE to false.");
//		System.out.println("-IGNORE_RUNS n1 n2 n3 -- List of run numbers to ignore - e.g. 2 4 5. " +
//				"(Optional (default: include all runs); only used in Block fMRI NPAIRS session files.)");
//		System.out.println("-SKIP_SCANS number -- The number of scans to skip at the beginning of each run" +
//				" (Optional (default: keep all scans); only used in Block fMRI NPAIRS session files).");
		System.out.println("-MERGE_ACROSS_RUNS true/false -- If true, merge data across all runs (default true); if " +
				"\nfalse, merge data within runs only. (Not used in Block NPAIRS.)");
		System.out.println("-CHANGE_DATA_PATH new_parent_path -- Include this flag and -ORIG_FILE_PREFIX to load an existing " +
			"\nsession file and change the parent data path to 'new_parent_path'. This option changes the path of the " +
			"\nparent data directory, i.e., the directory that contains the directory(ies) containing data. \nSee -ORIG_FILE_PREFIX.");
		System.out.println("-ORIG_FILE_PREFIX filename -- Session file to load. Must include -BLOCK flag (if true) when using this" +
				"\nflag. Use this flag to load existing session, make changes and save altered session file as " +
				"\n'[SESSION_FILE_DIR]SESSION_FILE_PREFIX'. Available changes: -DESCRIPTION, -BRAIN_MASK_FILE, " +
				"\n-USE_MASK_FILE, -MERGE_ACROSS_RUNS, -CHANGE_DATA_PATH.");
	}
	
	private static void printDatamatHelpMessage() {
		System.out.println("Create Datamat Usage:");
		System.out.println("-SESSION_FILE filename -- The session profile that will be used to create this datamat.");
		System.out.println("-COORD_THRESH x.xx -- Threshold to exclude voxels if no mask provided in session file (default 0.15).");
//		System.out.println("-IGNORE_SLICES n1 n2 n3 -- A list of slice numbers to ignore (default: include all of them).");
		System.out.println("-NORM_MEAN_VOL true/false -- If true, normalize data using the mean brain volume (default false).");
		System.out.println("-WINDOW_SIZE number -- The window (lag) size in no. of scans (Only required for Event-related analyses; default = 8)");
		System.out.println("-NORM_REF_SCANS true/false -- If true, normalize data using ref. scans (default true).");
		System.out.println("-SINGLE_SUBJECT true/false -- If true, format this datamat for single-subject analysis (default false).");
	}
	
	private static void printPlsSetupAnalysisHelpMessage() {
		System.out.println("PLS Create Analysis Setup Usage:");
//		System.out.println("NOTE: only PLSTYPE = 'MEAN-CENTERING_PLS' is currently available.");
		// Currently available: PLSTYPE = mean-centering_PLS, behaviour, non-rotated task, non-rotated behaviour
		System.out.println("-PLSTYPE type -- The analysis type (\"MEAN_CENTERING_PLS\", \"BEHAVIOR_PLS\", \"NON-ROTATED_TASK_PLS\", \"NON-ROTATED_BEHAVIOR_PLS\")");
//                      \"multiblock_PLS\")");
		System.out.println("-SETUP_FILE_PREFIX filename -- Prefix for the PLSAnalysisSetup file that will be created (automatically" +
				"\nappended with '_PLSAnalysisSetup.mat').");
		System.out.println("-SESSION_FILES grp1file1,grp1file2,grp1file3 grp2file1,grp2file2,grp2file3 -- The list of session files. Separate files in each group by " +
					"\ncommas and groups by spaces.");
//		System.out.println("-CONTRAST_FILENAME filename -- The contrast file to use if the analysis type is non-rotated task pls.");
//		System.out.println("-BEHAVIOR_FILENAME filename -- The behavior file to use if the analysis type is behavior pls or multiblock pls.");
//		System.out.println("-POSTHOC_FILENAME filename -- The post-hoc file to use if the analysis type is behavior pls or multiblock pls (optional).");
		System.out.println("-RESULTS_FILE_PREFIX filename -- Prefix for the results file.");
		System.out.println("-CONDITION_SELECTION n1 n2 n3 -- (default: include all conditions) Enter string of 1s and 0s separated by " +
				"\nwhitespace to indicate whether to include each condition - e.g. 1 0 1 1 to include 1st, 3rd and 4th condition" +
				"\nbut not 2nd. Condition order is taken from session files.");
//		System.out.println("-BEHAVIOR_BLOCK_CONDITION_SELECTION n1 n2 n3 -- Required for multiblock PLS; same syntax as for CONDITION_SELECTION.");
		System.out.println("-NUM_PERMUTATIONS n -- The number of permutations to run (default 0).");
		System.out.println("-NUM_BOOTSTRAPS n -- The number of bootstrap samples to run (default 0).");
		System.out.println("-CONFIDENCE_LEVEL int -- The confidence level for this analysis (default 95)." +
				"\n(0 < confid. level <= 100)");
	}
	
	private static void printNpairsSetupAnalysisHelpMessage() {
		System.out.println("NPAIRS Create Analysis Setup Usage:");
		System.out.println("\nRequired inputs:");
		System.out.println("-SESSION_FILES grp1file1,grp1file2 grp2file1,grp2file2... -- The list of session files. Separate " +
				"\nfiles within each group with commas; separate groups with spaces.");
		System.out.println("-RESULTS_FILE_PREFIX prefix -- Prefix for the results files (automatically appended with" +
				"\n'_NPAIRSJresult' + file-specific extensions).  Include full path.");
		System.out.println("\nOptional inputs:");
		System.out.println("-IGNORE_RUNS 1,3:0:1 2,4:1,2:0 -- Runs to be excluded from each input session file (1-relative). " +
				"\nSeparate runs within session file with commas; separate session files with colons; separate groups " +
				"\nwith spaces. If including this flag and no runs are to be excluded from given session file, enter '0'. " +
				"\n(Overrides 'ignore runs' info that may have been saved in input session files.)" +
				"\n\n**NOTE** 'ignore runs' info saved when generating NPAIRS Analysis Setup file via command line *WILL NOT* " +
				"\nbe recognized in NPAIRS Analysis GUI! Variable will be ignored and analysis setup file saved in results file" +
				"\ndirectory if running analysis via GUI will not include 'ignore_runs' variable! Run this analysis file via " +
				"\ncommand line.\n");
		System.out.println("-CLASS_SELECTION n1 n2 n3... -- Enter string of 1s and 0s separated by whitespace to " +
				"\nindicate whether to include each condition - e.g. 1 0 1 1 to include 1st, 3rd and 4th " +
				"\ncondition but not 2nd. Condition order is taken from session files (default: include all conditions).");
		System.out.println("-BLOCK true/false -- True if Blocked fMRI; false if Event-related (default true).");
		System.out.println("-DO_MSR true/false -- Remove mean session scan from data for each session before analysis " +
				"\n(default true).");
//		System.out.println("-DO_GLM true/false -- Do a General Linear Model analysis (default false). " +
//				"\n(Note that GLM has not been implemented yet so this flag must be set to false.)";
//		System.out.println("-DO_PCA true/false -- Do a Principal Components Analysis [before CVA] (default true)." +
//				"\n(Note that current implementation requires PCA + CVA to be run.)");
//		System.out.println("-DO_CVA true/false -- Do a Canonical Variance Analysis (default true)." +
//			"\n(Note that current implementation requires PCA + CVA to be run.)");
		System.out.println("-CVA_CLASSFILE filename -- File containing class info for the analysis. (Optional - default is " +
				"\nto get class info from session file condition info.) [Only applicable if you are doing Blocked " +
				"\nNPAIRS with CVA.]");
		System.out.println("-NORM_PCS true/false -- If true, normalize PC scores to have variance 1 before feeding into CVA " +
				"\n(default false).");
//		System.out.println("-DO_RESAMP true/false -- If true, resampling will be done during the analysis (default true)." +
//				"\n(Note that current implementation requires resampling to be done.)");
//		System.out.println("-RESAMP_TYPE resamp type -- The type of resampling to do.  Must be either \"split_half_xvalid\" +
//			"or \"bootstrap\". \n(Note that only split_half_xvalid is currently implemented.)");
		System.out.println("-NUM_SPLITS integer -- Upper bound on the number of times to split the data during resampling " +
				"\n(default 500).");
		System.out.println("-SPLIT_TYPE typename -- What to use as splitting unit when resampling data.  Can be SESSION or RUN " +
				"\n (default: SESSION). (Block fMRI only; split type must be SESSION for event-related fMRI NPAIRS.)");
		System.out.println("-SPLITS_INFO_FILENAME filename -- The file from which to load splits information (i.e., which data " +
				"\nvolumes belong in each split half.). Include this flag only if you want to use a previously " +
				"\ndetermined splitting schema instead of randomly generating splits for this analysis. (Mutually " +
				"\nexclusive with SPLIT_PARTITION/NUM_SPLITS option.)");
		System.out.println("-SPLIT_PARTITION partition1 partition2 -- How many session files to include in each split half. " +
				"\nTotal must be <= no. input session files. (Mutually exclusive with SPLITS_INFO_FILENAME option.) " +
				"\n(default: split session files as evenly as possible; in case no. session files N is odd, partition " +
				"\nwill be {(N-1)/2, (N+1)/2})");
//		System.out.println("-DO_INIT_EVD true/false -- If true, reduce the size of input data before analysis via an eigenvalue decomposition (default true).");
//		System.out.println("-LOAD_EVD true/false -- If true, load the initial eigenvalue decomposition from a file; otherwise calculate decomposition from " +
//					"scratch (default false).");
		System.out.println("-EVD_FILE_PREFIX prefix -- The prefix (including full path) for the file from which to load " +
				"\ninitial eigenvalue decomposition. Include this flag only if you want to load the EVD from " +
				"\nexisting files (default is to do it from scratch and save it). EVD files saved in an NPAIRS " +
				"\nanalysis have format prefix_NPAIRSJresult.evals/.evects.");
		System.out.println("-DRF decimal number -- The data reduction factor to use (0.0 < DRF <= 1.0). DRF is the proportion " +
				"\nof data dimensions to keep after EVD and pass on to the next analysis step (default = 0.3).");
		System.out.println("-SET_PC_RANGE r1-r2 s m -- Set a range of how many PCs to pass into CVA after PCA. A new NPAIRS " +
				"\nanalysis is run for each no. of PCs.  Syntax: r1-r2 (no spaces around hyphen) s m where r1 is min " +
				"\nno. PCs to use in split data analysis, r2 is max no. PCs for split data, s is no. of PCs by which to " +
				"\nincrease split data range at each step & m is ratio of no. PCs to use in full data analysis (relative " +
				"\nto no. PCs for split data). " +
				"\nE.g. 10-20 5 2.0 ==> run 3 analyses: " +
				"\n(1) PCs (split data) = 1-10 & PCs (full data) = 1-20;" +
				"\n(2) PCs (split data) = 1-15 & PCs (full data) = 1-30;" +
				"\n(3) PCs (split data) = 1-20 & PCs (full data) = 1-40.");
		System.out.println("-PCS_SPLIT range -- PCs to use for split data, e.g. 1-4,6,10-30 (no whitespace around hyphens). " +
				"\nIgnored if SET_PC_RANGE set.");
		System.out.println("-PCS_ALL_DATA range -- PCs to use for full data, e.g. 1-10,15,25-30 (no whitespace around hyphens). " +
				"\nIgnored if SET_PC_RANGE set.");
		System.out.println("(NOTE: default if neither SET_PC_RANGE nor PCS_ALL_DATA/PCS_SPLIT are set: set PC range " +
				"\nautomatically to max possible split range 2-SPLITMAX, step size 1, for both split and full-data " +
				"\nanalyses)");
		System.out.println("-SAVE_LOTS true/false -- If true, save additional result files besides the summary .mat file " +
				"\n(default false).");
		System.out.println("-SAVE_SPLITS true/false -- If true and SAVE_LOTS is true, save results from each split analysis " +
				"\n(default false).");
		System.out.println("-SETUP_FILE_PREFIX prefix -- Prefix for analysis setup file (automatically appended with either " +
				"\n'_NPAIRSAnalysisSetup.mat' [Blocked fMRI] or '_NPAIRSAnalysisSetupER.mat' [Event-related fMRI]. " +
				"\nInclude full path." +
				"\n(default: use results file prefix as setup file prefix and save setup file in results directory)");
	}
	
	private static void printRunAnalysisHelpMessage() {
		System.out.println("Run Analysis Usage:");
		System.out.println("filename1 filename2 filename3 -- A list of analysis setup files to run.");
		System.out.println("(NPAIRS ONLY) Optional final flag after analysis filenames - one of (a) or (b):" +
				"\n(a) EVD_ONLY (to run and save only initial EVD before exiting program without running analyses)" +
				"\nOR" +
				"\n(b) X (to run only Xth analysis, where X is an integer between 1 and M, given PC range consisting" +
				"\nof M total analyses. Required: initial EVD has already been run and saved in results directory.)"); 
				
		
	}
}