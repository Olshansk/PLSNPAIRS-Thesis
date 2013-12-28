package pls.analysis;

import pls.chrome.shared.ProgressDialogWatcher;
import pls.shared.GlobalVariablesFunctions;
import pls.shared.NpairsAnalysisSetupERFileFilter;
import pls.shared.NpairsAnalysisSetupFileFilter;
import pls.shared.NpairsfMRIResultFileFilter;
import npairs.Npairsj;
import npairs.io.NpairsDataLoader;
import npairs.io.NpairsjIO;
import npairs.NpairsjSetupParams;
import npairs.NpairsjException;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;

import java.util.Date;
import java.text.DateFormat;
import java.text.SimpleDateFormat;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;


public class NpairsAnalysis extends ProgressDialogWatcher {
	
	String npairsSetupParamsMatFileName;
	String matlibType;
	String matlibTypeForInitFeatSel;
	//Alan
	String hadoopDirectory = "/user/" + System.getProperty("user.name") + "/";
	
	int numNPAIRS = 1;
	
	boolean isBlocked;     // set to false for event-related fMRI
	boolean evdOnly;       // if true then just run initial EVD, save it and exit. Don't do analysis.
	int analysisNum = -1;  // if set to positive integer then run only that analysis number (given
	                       // multiple analyses that can be run from given setup file, e.g. when
	                       // pc range is set in setup file)
	boolean loadEVD;       // Overrides nsp.loadEVD.  Set to true if analysisNum is set ( > 0) 
	                       // because then we assume that EVD has already been run and saved,
	                       // even if nsp.loadEVD = false.
	
	public NpairsAnalysis(String npairsSetupParamsMatFileName, String matlibType, 
			String matlibTypeForInitFeatSel, boolean isBlocked, boolean evdOnly, int analysisNum) {
		this.npairsSetupParamsMatFileName = npairsSetupParamsMatFileName;
		this.matlibType = matlibType;
		this.matlibTypeForInitFeatSel = matlibTypeForInitFeatSel;
		this.isBlocked = isBlocked;
		this.evdOnly = evdOnly;
		this.analysisNum = analysisNum;
	}
	
	public NpairsAnalysis(String npairsSetupParamsMatFileName, String matlibType, 
			String matlibTypeForInitFeatSel, boolean isBlocked) {
		this(npairsSetupParamsMatFileName, matlibType, matlibTypeForInitFeatSel, isBlocked, false, -1);
	}
	
	public void doTask() throws Exception {
		String dataSource = "image files";
		String analysisType = "blocked";	
		if (!isBlocked) {
			dataSource = "datamats";
			analysisType = "event-related";
		}

		String versionMessage = "Using plsnpairs version # " + 
			GlobalVariablesFunctions.getVersion() + ".\n";
		String libTypeMessage = "Using matrix library type: " + matlibType + ".\n";
		String analysisTypeMessage = "Doing " + analysisType + " fMRI analysis.\n";
		String evdOnlyMessage = "Doing initial EVD only. \nRunning and saving initial EVD.\n";
		String evdOnlyLoadEVDMessage = "Doing initial EVD only. \nEVD to be loaded from files.";
		String setupFileIDMessage = "Running NPAIRS setup file...";

		progress.postMessage(versionMessage);
		progress.postMessage(libTypeMessage);
	
		
		progress.startTask(setupFileIDMessage, "All tasks");
		
		NpairsjSetupParams nsp = new NpairsjSetupParams(npairsSetupParamsMatFileName, !isBlocked);
		
		File nspFileField = new File(nsp.resultsFilePrefix.trim());
		File resultsDir = nspFileField.getParentFile();
		String resultFileName = nspFileField.getName();
		
		if (resultsDir == null) {
			// set results dir to current working directory if none specified
			resultsDir = new File(System.getProperty("user.dir")); 
		}
		
		// create results dir if it doesn't exist
		if (!resultsDir.exists()) {
			String createDirMess = "Creating results directory " + resultsDir.toString();
			progress.postMessage(createDirMess + "\n");
			Npairsj.output.println(createDirMess);
			boolean created = resultsDir.mkdir();
			if (!created) {
				String errorMessage = "Unable to create results directory " + resultsDir + ".";
				progress.postMessage(errorMessage + "\n");
				Npairsj.output.println(errorMessage);
				
				progress.endTask();
				progress.complete();
				return;
			}
		}
		
		if (!resultsDir.canWrite()) {
			String errorMessage = "Unable to save files in results directory " + resultsDir + ".";
			progress.postMessage(errorMessage + "\n");
			Npairsj.output.println(errorMessage);
		}
		
		else {

			loadEVD = nsp.loadEVD;

			if (evdOnly) {
				if (!loadEVD){
					progress.postMessage(evdOnlyMessage);
				}
				else {
					progress.postMessage(evdOnlyLoadEVDMessage);
				}
			}
			else {
				progress.postMessage(analysisTypeMessage);
			}

			nsp.initLogFile(evdOnly, analysisNum); // sets Npairsj.output to log file in results dir
			Npairsj.output.println(getDateTime());
			Npairsj.output.println(versionMessage);
			Npairsj.output.println(libTypeMessage);
			if (evdOnly) {
				if (!loadEVD) {
					Npairsj.output.println(evdOnlyMessage);
				}
				else Npairsj.output.println(evdOnlyLoadEVDMessage);
			}
			else {
				Npairsj.output.println(analysisTypeMessage);
			}
			Npairsj.output.println(setupFileIDMessage);

			try {
				//Copy the setup file into the result directory.
				String ext;
				if(isBlocked){
					ext = NpairsAnalysisSetupFileFilter.ext;
				}else{
					ext = NpairsAnalysisSetupERFileFilter.ext;
				}
				String saveSetupMess = "Saving setup file " + nsp.resultsFilePrefix + ext
					+ "... ";
				progress.postMessage(saveSetupMess + "\n");
				Npairsj.output.println(saveSetupMess);
				NpairsjIO.copyFile(npairsSetupParamsMatFileName, 
						resultsDir.getAbsolutePath() 
						+ File.separator + resultFileName + ext);
			}
			catch (Exception e) {
				String errorMessage = "Unable to save copy of analysis file into results " +
				"directory.";
				progress.postMessage(errorMessage + "\n");
				Npairsj.output.println(errorMessage);

			}
			//TODO: warn user about permissions and let the user choose to stop or continue analysis
			//     String evdStatusMessage = "";
			NpairsDataLoader ndl = null;

			if (!(evdOnly && loadEVD)) {
				// if evdOnly and loadEVD, program will exit immediately; otherwise 
				// this part is executed
				String dataSourceMessage = "Loading data from " + dataSource + "...";
				progress.startTask(dataSourceMessage, "Data loading/EVD");
				Npairsj.output.println(dataSourceMessage);
				if (nsp.initFeatSelect) {
					String ifsMsg = "Doing initial feature selection ";
					if (nsp.normedEVD) {
						ifsMsg = ifsMsg.concat("(normed EVD)...\n");
					}
					else if (nsp.initEVD) {
						ifsMsg = ifsMsg.concat("(EVD)...\n");
					}
					progress.postMessage(ifsMsg);
					if (analysisNum > 0) {
						loadEVD = true; // assume EVD already run and saved if specifying which
						// analysis to run
						nsp.setEVDLoad(true); 
						nsp.setEVDFilePrefix(); // sets EVD file prefix to results file prefix (if not already set)
					}
					//        		else {
					//        			loadEVD = nsp.loadEVD;
					//        		}

					if (loadEVD) {
						progress.postMessage("Loading EVD from files.\n");
					}	
					else {
						String libTypeEVDMessage = "Using matrix library type: " + matlibTypeForInitFeatSel + 
						" to do initial feature selection.\n";
						progress.postMessage(libTypeEVDMessage);
						Npairsj.output.println(libTypeEVDMessage);
					}
				}
				ndl = new NpairsDataLoader(nsp, matlibType, 
						matlibTypeForInitFeatSel, analysisType.equals("event-related"));
				progress.endTask();
				//   	evdStatusMessage = "Finished running and saving initial EVD.";
			}


			if (evdOnly) {
				String evdDoneMessage = "Exiting program without running analysis.\n";
				progress.postMessage(evdDoneMessage);
				Npairsj.output.println(evdDoneMessage);
				return;
			}

			// Run NPAIRS
			checkMultNPAIRS(nsp);
			if (numNPAIRS == 1) {
				progress.startTask("Running NPAIRS analysis\n", "Analysis");
			} 
			else if (analysisNum == -1) { // run all N analyses (where N > 1)
				progress.postMessage("Starting NPAIRS analysis loop.\n");
			} 

			int firstAnalysisIdx = 0;
			int lastAnalysisIdx = numNPAIRS - 1;
			if (analysisNum > 0) { // just run the analysisNum'th analysis
				firstAnalysisIdx = analysisNum - 1;
				lastAnalysisIdx = analysisNum - 1;
			}
			long start=System.currentTimeMillis();

			for (int i = firstAnalysisIdx; i <= lastAnalysisIdx; ++i) {
				if (numNPAIRS > 1) {
					String runMsg = "Running analysis # " + (i + 1) + "... ";
					progress.startTask(runMsg, 
							"Analysis # " + (i + 1));
					Npairsj.output.println(runMsg);
				}

				// fill in required setup parameters for current analysis
				boolean abortAnalyses = false;
				boolean pcsOutOfRange = false;
				try {
					pcsOutOfRange = nsp.setPCs(i, progress);
					if (pcsOutOfRange && i < lastAnalysisIdx) {
						//  pc range increases in each analysis so all subsequent analyses 
						// will also be out of range
						abortAnalyses = true;
						lastAnalysisIdx = i;
					}
				}
				catch (NpairsjException npe) { // no pcs in range
					String errMsg = npe.getMessage() + " Stopping analysis.";
					Npairsj.output.println(errMsg);
					progress.postMessage("\n" + errMsg + "\n");
					progress.endTask();
					progress.complete();
					return;
				}
				if (nsp.setPCrange) {
					nsp.setResultsFilePref(i);
				}

				//				}
				//				else {
				//					if (!nsp.splitPCRangeValid()) {
				//						String splitPCRangeWarning = "Warning: PC Range for split data analyses" +
				//						"\nexceeds size of data.  Out-of-range PCs have been excluded.";
				//						progress.postMessage("\n" + splitPCRangeWarning + "\n");
				//						Npairsj.output.println(splitPCRangeWarning);
				//					}
				//					if (!nsp.fullPCRangeValid()) {
				//						String fullPCRangeWarning = "Warning: PC Range for full data analyses" +
				//						"\nexceeds size of data.  Out-of-range PCs have been excluded.";
				//						progress.postMessage("\n" + fullPCRangeWarning + "\n");
				//						Npairsj.output.println(fullPCRangeWarning);
				//					}
				//
				//				}

				try {
					if(i == firstAnalysisIdx){
						System.out.println("starting to serialized dataloader object");

						try
						{	
							ObjectOutput out = new ObjectOutputStream(new FileOutputStream("dataLoader.ser"));
						    out.writeObject(ndl);
						    out.close();				
						}
						catch(IOException ex)
						{
							ex.printStackTrace();
						}
						 FileSystem hdfsFileSystem = FileSystem.get(new Configuration());
						    Path local = new Path("dataLoader.ser");
						    Path hdfs = new Path(hadoopDirectory); 
						    hdfsFileSystem.copyFromLocalFile(true, local, hdfs);
					}
					Npairsj npairsj = new Npairsj(ndl, nsp, matlibType);

					String saveResultMatMessage = "Saving NPAIRS results to file " + 
					nsp.resultsFilePrefix + NpairsfMRIResultFileFilter.EXTENSION + "...";
					progress.postMessage(saveResultMatMessage);
					Npairsj.output.print(saveResultMatMessage);
					double sTime = System.currentTimeMillis();
					new ResultSaver(npairsj, npairsSetupParamsMatFileName, nsp.resultsFilePrefix);
					double tTime = (System.currentTimeMillis() - sTime) / 1000;
					progress.postMessage("Done [" + tTime + "s]\n");
					Npairsj.output.println("Done [" + tTime + "s]");

					
					//Alan:
					npairsj = null;
				}
				catch (NpairsjException npe) {
					progress.printError(npe.getMessage());
					Npairsj.output.println(npe.getMessage());
				}
				progress.endTask(); // running current analysis
				
				if (abortAnalyses) {
					String stopAnalysesMsg = "Not running remaining analyses - PCs out of range.";
					progress.postMessage("\n" + stopAnalysesMsg + "\n");
					Npairsj.output.println(stopAnalysesMsg);
				}				
			}	
			long end=System.currentTimeMillis();
			System.out.println("THE ANALYSIS TIME IS"+(end-start)/1000+" seconds ********************");
			
			//Alan
			System.gc();
		}
		progress.endTask();
		progress.complete();
	}
	

	private void checkMultNPAIRS(NpairsjSetupParams nsp) throws NpairsjException, 
		IOException {
		
		numNPAIRS = nsp.numNPAIRS;
		if (numNPAIRS > 1) {
			progress.postMessage("Setup file contains " + numNPAIRS + " NPAIRS analyses.\n");
		}	
	}
			
    public static String getDateTime() {
        DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
        Date date = new Date();
        return dateFormat.format(date);
    }
	
//	public void finalize() throws Throwable {
//		System.out.println("NpairsjAnalysis killed.");
//		
//		super.finalize();
//	}
	
}