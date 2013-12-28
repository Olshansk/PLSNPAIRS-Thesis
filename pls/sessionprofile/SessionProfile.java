package pls.sessionprofile;

import java.io.IOException;
import java.util.ArrayList;
import java.util.NoSuchElementException;
import java.util.Vector;
import java.util.Iterator;

import javax.swing.JOptionPane;

import pls.chrome.sessionprofile.ChangeDataPathsFrame;
import pls.chrome.sessionprofile.SessionProfileFrame;
import pls.shared.BfMRISessionFileFilter;
import pls.shared.MLFuncs;
import pls.shared.NpairsBlockSessionFileFilter;
import pls.shared.NpairsERSessionFileFilter;
import pls.shared.fMRISessionFileFilter;

import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLStructure;

import extern.NewMatFileReader;

public class SessionProfile {
	public String description;
	public String datamatPrefix;
	
	public boolean mergeAcrossRuns;
	
	public Vector<String[]> conditionInfo;
	public Vector<RunInformation> runInfo;
	
	public boolean useBrainMask = false;
	public String brainMaskFile = "";
	public boolean useAllVox = false; 
	public double thresh = 0.15;
	
	public boolean isNPAIRS;
	
	public boolean isBlock;
	
	public int numSkippedScans;
	
	public int[] ignoreRuns;
	
	public SessionProfile() {}
	
	public SessionProfile(String description, String datamatPrefix, boolean mergeAcrossRuns,
			Vector<String[]> conditionInfo, Vector<RunInformation> runInfo, boolean useBrainMask,
			String brainMaskFile, boolean isNPAIRS, boolean isBlock, int numSkippedScans,
			int[] ignoreRuns, boolean useAllVox) {
		this.description = description;
		this.datamatPrefix = datamatPrefix;
		this.mergeAcrossRuns = mergeAcrossRuns;
		this.conditionInfo = conditionInfo;
		this.runInfo = runInfo;
		this.useBrainMask = useBrainMask;
		this.brainMaskFile = brainMaskFile;
		this.useAllVox = useAllVox;
		this.isNPAIRS = isNPAIRS;
		this.isBlock = isBlock;
		this.numSkippedScans = numSkippedScans;
		this.ignoreRuns = ignoreRuns;
	}
	
	public static SessionProfile loadSessionProfile(String fileName, boolean isBlockedFmri, boolean isNpairs) throws IOException {
		SessionProfile sessionProfile = new SessionProfile();
		
		sessionProfile.isNPAIRS = isNpairs;
		sessionProfile.isBlock = isBlockedFmri;
		
		// add extension if missing from fileName
		String extension = "";
		if (sessionProfile.isNPAIRS && sessionProfile.isBlock) {
			extension = NpairsBlockSessionFileFilter.EXTENSION;	
		}
		else if (sessionProfile.isBlock) {
			extension = BfMRISessionFileFilter.EXTENSION;
		}
		else {
			extension = fMRISessionFileFilter.EXTENSION;
		}
		int extInd = fileName.indexOf(extension);
		if (extInd == -1) { // no extension; must add one
			extInd = fileName.length() + 1;
			fileName += extension;
		}
		
		// Get needed variables from file
		MLStructure sessionInfo = null;
		
		sessionInfo = (MLStructure)new NewMatFileReader(fileName).getContent().get("session_info");
		
		sessionProfile.description = ((MLChar)sessionInfo.getField("description")).getString(0);
		
		// datamat not used in npairs but included by default for compatibility with PLS
		sessionProfile.datamatPrefix = ((MLChar)sessionInfo.getField("datamat_prefix")).getString(0);
		
		
		sessionProfile.mergeAcrossRuns = true;
		
		if (((MLDouble)sessionInfo.getField("across_run")) != null) {
			sessionProfile.mergeAcrossRuns = ((MLDouble)sessionInfo.getField("across_run")).get(0, 0).intValue() == 1;
		}
		
		String[] conditions;
		MLCell conditionBaseline;
		
		// If there is no condition0, try to get the information from condition
		// Note that this check is only necessary for some hand-made session profiles.
		if ((MLCell)sessionInfo.getField("condition0") != null) {
			conditions = MLFuncs.MLCell1dRow2StrArray((MLCell)sessionInfo.getField("condition0"));
		} else {
			conditions = MLFuncs.MLCell1dRow2StrArray((MLCell)sessionInfo.getField("condition"));
		}
		
		// If there is no condition_baseline0, try to get the information from condition_baseline
		// Note that this check is only necessary for some hand-made session profiles.
		if ((MLCell)sessionInfo.getField("condition_baseline0") != null) {
			conditionBaseline = (MLCell)sessionInfo.getField("condition_baseline0");
		} else {
			conditionBaseline = (MLCell)sessionInfo.getField("condition_baseline");
		}
		
		sessionProfile.conditionInfo = new Vector<String[]>();
	    for(int i = 0; i < conditions.length; i++) {
	    	String conditionName = conditions[i];
	    	MLDouble baseline = (MLDouble)conditionBaseline.get(0, i);
	    	String refScanOnset = new Integer(baseline.get(0, 0).intValue()).toString();
	    	String numRefScans = new Integer(baseline.get(0, 1).intValue()).toString();
	    	sessionProfile.conditionInfo.add(new String[]{conditionName, refScanOnset, numRefScans});
	    }
		
		sessionProfile.runInfo = SessionProfileFrame.getRunInformation(
				(MLStructure)sessionInfo.getField("run"), isBlockedFmri);
		
		// mask info
		sessionProfile.useAllVox = false;
		if (((MLDouble)sessionInfo.getField("use_all_voxels")) != null) {
			sessionProfile.useAllVox = (((MLDouble)sessionInfo.getField("use_all_voxels")).get(
					0, 0).intValue() == 1);	
		}
		
		if (!sessionProfile.useAllVox) {
			if (sessionInfo.getField("mask") != null) {
				sessionProfile.useBrainMask = true;
				sessionProfile.brainMaskFile = ((MLChar)sessionInfo.getField("mask")).getString(0);
			}
			else {
				try {
					sessionProfile.thresh = ((MLDouble)sessionInfo.getField("brain_coord_thresh")).
						get(0);
				}
				catch (NullPointerException npe) {
					// threshold info wasn't being saved in session file yet when this one
					// was created
				}
			}
		}
		
		if (sessionInfo.getField("scans_skipped") != null) {
			sessionProfile.numSkippedScans = ((MLDouble)sessionInfo.getField("scans_skipped")).get(0,0).intValue();
		}
		
		if (sessionInfo.getField("runs_skipped") != null) {
			String[] sIgnoreRuns = ((MLChar)sessionInfo.getField("runs_skipped")).getString(0).split(" ");
			if(sIgnoreRuns[0].length() != 0) {
				sessionProfile.ignoreRuns = new int[sIgnoreRuns.length];
				for(int i = 0; i < sIgnoreRuns.length; i++) {
					sessionProfile.ignoreRuns[i] = Integer.parseInt(sIgnoreRuns[i]);
				}
			}
		}
		
		return sessionProfile;
	}
	
	/** Merges runs from each input session profile into a single session profile.
	 *  Uses mask information from first input session profile
	 * @param fileNames 
	 * 			files containing session profiles to merge
	 * @param description
	 * 			description for new merged session file
	 * @param save name of merged session file
	 * REQUIRED: all input session files have same conditions and are same type of
	 *  session file (Block or Event-related) for same type of analysis (NPAIRS or PLS)
	 */
	public static void mergeSessionProfiles(String[] fileNames, String description, 
			String saveName, boolean isBlockedFmri, boolean isNpairs) throws IOException {
		int nSessProfs = fileNames.length;
		SessionProfile[] sessProfs = new SessionProfile[nSessProfs];
		
		SessionProfile mergedSessProf = loadSessionProfile(fileNames[0], isBlockedFmri,
					isNpairs);
		int nConds = mergedSessProf.conditionInfo.size();
		for (int s = 1; s < nSessProfs; ++s) {
			sessProfs[s] = loadSessionProfile(fileNames[s], isBlockedFmri, isNpairs);
			// check conditions
			if (sessProfs[s].conditionInfo.size() != nConds) {
				throw new IllegalArgumentException("Error - input session files must have " +
						"same number of conditions");
			}
			for (int c = 0; c < nConds; ++c) {
				for (int i = 0; i < 3; ++i) {
					if (!(sessProfs[s].conditionInfo.get(c)[i].equals(
							mergedSessProf.conditionInfo.get(c)[i]))) {
						throw new IllegalArgumentException("Error - input session files " +
								"must have same conditions and reference scan info");
					}
				}
			}
			
			// merge run info
			Iterator<RunInformation> runIter = sessProfs[s].runInfo.iterator();
			while (runIter.hasNext()) {
				mergedSessProf.runInfo.add(runIter.next());
			}		
		}
		
		mergedSessProf.description = description;
//		System.out.println("Saving merged SP...");
		mergedSessProf.saveSessionProfile(saveName);
	}
	
	
	public void saveSessionProfile(String fileName) throws IOException {
		MLStructure sessionInfo = new MLStructure("session_info", new int[] {1, 1});
		
		sessionInfo.setField("description", new MLChar("description", description) );
		
		// Save pls_data_path and other pls-specific variables even in npairs session file
		// so that session files are compatible.
		// TODO should also save npairs-specific variables (i.e., skipped scans and runs info)
		// in java-generated pls session files; that way, they will be totally compatible with
		// npairs, but should still be backwards compatible with plsgui in matlab, since extra
		// npairs variables should just be ignored when read in by plsgui
		
		try {
			String dataPath = ChangeDataPathsFrame.getCommonPath(runInfo);
			sessionInfo.setField("pls_data_path", new MLChar("pls_data_path", dataPath));
		}
		catch (NullPointerException npe) {
			throw new NullPointerException("Unable to find data directory " +
					runInfo.get(0).dataDirectory + ".");
		}
		String dmPrefix = null;
		// add extension if required to filename
		String extension = "";
		if (isNPAIRS && isBlock) {
			extension = NpairsBlockSessionFileFilter.EXTENSION;	
		}
		else if (isBlock) {
			extension = BfMRISessionFileFilter.EXTENSION;
		}
		else {
			extension = fMRISessionFileFilter.EXTENSION;
		}
		
		int extInd = fileName.indexOf(extension);
		if (extInd == -1) { // no extension; must add one
			extInd = fileName.length() + 1;
			fileName += extension;
		}
		try {
			dmPrefix = datamatPrefix;
		}
		catch (NullPointerException npe) {
			// it's an NPAIRS Block analysis; create default datamat prefix using session 
			// file save prefix
			int parentEnd = fileName.lastIndexOf(System.getProperty("file.separator"));
			dmPrefix = fileName.substring(parentEnd + 1, extInd - 1);	
		}

		sessionInfo.setField("datamat_prefix", new MLChar("datamat_prefix", dmPrefix));

		// set condition info for single run ('num_conditions0', 'conditions0', 'condition_baseline0')
        int numConds = conditionInfo.size();
        int numRuns = runInfo.size();

        MLCell conditions0 = new MLCell("conditions0", new int[] {1, numConds});
        MLCell conditionBaseline0 = new MLCell("condition_baseline0", new int[] {1, numConds});
        MLDouble numConditions0 = new MLDouble("num_conditions0", new double[][] {{numConds}});
        
        for(int i = 0; i < numConds; i++) {
        	String conditionName = conditionInfo.get(i)[0];
        	double refScanOnset = new Double(conditionInfo.get(i)[1]).doubleValue();
        	double numRefScans = new Double(conditionInfo.get(i)[2]).doubleValue();
        	conditions0.set(new MLChar("condition0" + i, conditionName), 0, i);
        	conditionBaseline0.set(new MLDouble("condition_baseline0" + i, 
        			new double[][]{{refScanOnset, numRefScans}}), 0, i);
        }

        sessionInfo.setField("num_conditions0", numConditions0);
        sessionInfo.setField("condition0", conditions0);
        sessionInfo.setField("condition_baseline0", conditionBaseline0);
        
        
        // set condition info for all conditions/runs ('num_conditions', 'conditions', 'condition_baseline')
        MLCell conditions = conditions0;
        MLCell conditionBaseline = conditionBaseline0;
        MLDouble numConditions = numConditions0;
        
        if (!mergeAcrossRuns) {
        	// 'num_conditions', 'conditions' and 'condition_baseline' treat same condition
        	// in different runs as separate conditions
        	numConditions = new MLDouble("num_conditions", new double[][] {{numConds * numRuns}});
        	conditions = new MLCell("conditions", new int[] {1, numRuns * numConds});
        	conditionBaseline = new MLCell("condition_baseline", new int[] {1, numRuns * numConds});
        	int count = 0;
        	for(int i = 0; i < numRuns; i++) {
        		for(int j = 0; j < numConds; j++, count++) {
        			String conditionName = "Run" + (i + 1) + conditionInfo.get(j)[0];
        			double refScanOnset = new Double(conditionInfo.get(j)[1]).
        			doubleValue();
        			double numRefScans = new Double(conditionInfo.get(j)[2]).
        			doubleValue();
        			conditions.set(new MLChar("condition" + count, conditionName), 0, count);
        			conditionBaseline.set(new MLDouble("conditions_baseline" + count, 
        					new double[][]{{refScanOnset, numRefScans}}), 0, count);
        		}
        	}
        }
        
        sessionInfo.setField("num_conditions", numConditions);
        sessionInfo.setField("condition", conditions);
        sessionInfo.setField("condition_baseline", conditionBaseline);
        
	    // set run info
        sessionInfo.setField("num_runs", new MLDouble("num_runs", new double[][]{{numRuns}}));
		
		MLStructure run = new MLStructure("run", new int[]{1, numRuns});
		for(int i = 0; i < numRuns; i++) {
			String[] files = runInfo.get(i).dataFiles.split(" ");
			int ns = files.length;
			run.setField("num_scans", new MLDouble("num_scans", new double[][]{{ns}}), i);
			run.setField("data_path", new MLChar("data_path", runInfo.get(i).
					dataDirectory), i);
			MLCell cFiles = new MLCell("data_files" + i, new int[]{files.length, 1});
			for(int j = 0; j < files.length; j++) {
				cFiles.set(new MLChar("data_file" + j, files[j]), j, 0);
			}
			run.setField("data_files", cFiles, i);
			
			int extensionPos = files[0].lastIndexOf(".");
			run.setField("file_pattern", new MLChar("file_pattern", "*" + files[0].substring(extensionPos)));
			
			String onsetVariableName = "evt_onsets";
			if(isBlock) {
				onsetVariableName = "blk_onsets";
			}
			
			MLCell cEventOnsets = new MLCell(onsetVariableName + i, new int[]{1, numConds});
			for(int j = 0; j < numConds; j++) {
				double[][] dEventOnsets = null;
				String[] sEventOnsets = runInfo.get(i).onsets.get(j).split(" ");
				for(String s : sEventOnsets) {
					dEventOnsets = MLFuncs.append(dEventOnsets, 
							new double[][]{{new Double(s).doubleValue()}});
				}
				cEventOnsets.set(new MLDouble(onsetVariableName + j, dEventOnsets), 0, j);
			}
			run.setField(onsetVariableName, cEventOnsets, i);
			
			if(isBlock) {
				MLCell cEventLengths = new MLCell("blk_length" + i, new int[]{1, numConds});
				for(int j = 0; j < numConds; j++) {
					double[][] dEventLengths = null;
					String[] sEventLengths = runInfo.get(i).lengths.get(j).split(" ");
					for(String s : sEventLengths) {
						dEventLengths = MLFuncs.append(dEventLengths, 
								new double[][]{{new Double(s).doubleValue()}});
					}
					cEventLengths.set(new MLDouble("blk_length" + j, dEventLengths), 0, j);
				}
				run.setField("blk_length", cEventLengths, i);
			}
		}
		sessionInfo.setField("run", run);
		

		if (mergeAcrossRuns) {
			 // (recall: 'across_run' is set to 1 by default in NPAIRS session file)
			sessionInfo.setField("across_run", new MLDouble("across_run", new double[][]{{1}}));
		} 
		else {
				sessionInfo.setField("across_run", new MLDouble("across_run", 
						new double[][]{{0}}));
		}
		
		if	(useBrainMask) {
			sessionInfo.setField("mask", new MLChar("mask", brainMaskFile) );
		}
		
		// add datamat setup info to session file (required for npairs)
		sessionInfo.setField("scans_skipped", new MLDouble("scans_skipped", 
				new double[][] {{numSkippedScans}}));
		String skippedRunsText = "";
		if (ignoreRuns != null) {
			for(int i : ignoreRuns) {
				skippedRunsText += i + " ";
			}
			if (skippedRunsText.endsWith(" ") ) {
				skippedRunsText = skippedRunsText.substring(0, skippedRunsText.length() - 1);
			}
		}
		if (skippedRunsText.equals("")) {
			skippedRunsText = "0";
		}
		sessionInfo.setField("runs_skipped", new MLChar("runs_skipped", skippedRunsText));
				
		ArrayList<MLArray> list = new ArrayList<MLArray>();
		list.add(sessionInfo);
        list.add(new MLChar("create_ver", "999999"));
        
    	new MatFileWriter(fileName, list);
	}
	
	public int getNumInclRuns() {
		int nIgnore = ignoreRuns.length;
		if (nIgnore == 1 && ignoreRuns[0] == 0) {
			nIgnore = 0;
		}
		return runInfo.size() - nIgnore;
	}

	public void changeDataPath(String newDataPath) {
		for (RunInformation r : this.runInfo) {
			String[] pathArray = r.dataDirectory.split("/");
			pathArray = pathArray[pathArray.length - 1].split("\\\\");
			r.dataDirectory = newDataPath + System.getProperty("file.separator") + pathArray[pathArray.length - 1];
		}
	}
	
	public void setDescription(String description) {
		this.description = description;
	}
	
	public void setDmatPrefix(String dmatPref) {
		this.datamatPrefix = dmatPref;
	}
	
	public void setBlock(boolean isBlock) {
		this.isBlock = isBlock;
	}
	
	public void setUseMask(boolean useMask) {
		this.useBrainMask = useMask;
	}
	
	public void setMaskFile(String maskFile) {
		this.brainMaskFile = maskFile;
	}
	
	public void setIgnoreRuns(int[] runsToIgnore) {
		this.ignoreRuns = runsToIgnore;
	}
	
	public void setNumSkippedScans(int numSkipped) {
		this.numSkippedScans = numSkipped;
	}
	
	public void setMergeAcrossRuns(boolean merge) {
		this.mergeAcrossRuns = merge;
	}
	
	public void setConditionInfo(Vector<String[]> condInfo) {
		this.conditionInfo = condInfo;
	}
	
	public void setDataFiles(String[] dataFiles) {
		// TODO: set data files in runInfo.
		System.out.println("SessionFile.setDataFiles(dataFiles) not implemented yet...");
	}
	
	public void setDataPaths(String[] dataPaths) {
		// TODO: set data paths in runInfo.
		System.out.println("SessionFile.setDataPaths(dataPaths) not implemented yet...");
	}
	
	public void setOnsets(String[] onsets) {
		// TODO: set data paths in runInfo.
		System.out.println("SessionFile.setOnsets(onsets) not implemented yet...");
	}
	
}
