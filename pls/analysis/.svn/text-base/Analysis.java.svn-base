package pls.analysis;

import java.util.Map;
import java.util.Vector;

import com.jmatio.types.MLArray;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLStructure;

import extern.NewMatFileReader;
import pls.chrome.shared.ProgressDialogWatcher;
import pls.shared.MLFuncs;

public class Analysis extends ProgressDialogWatcher {
	
	private int imagingType = 0;
	private int plsType = 0;
	private Vector<String[]> sessionProfiles = null;
	private String contrastFilename = null;
	private String behaviorFilename = null;
	private String resultsFilename = null;
	private int[] conditionSelection = null;
	private int[] behaviorBlockConditionSelection = null;
	public int numPermutations = 0;
	public int numBootstraps = 0;
	private double confidenceLevel = 0;
	

	public Analysis(int imagingType, int plsType, Vector<String[]> sessionProfiles, 
			String contrastFilename, String behaviorFilename, 
			String resultsFilename, int[] conditionSelection, 
			int[] behaviorBlockConditionSelection, int numPermutations, int numBootstraps, 
			double confidenceLevel) {
		this.imagingType = imagingType;
		this.plsType = plsType;
		this.sessionProfiles = sessionProfiles;
		this.contrastFilename = contrastFilename;
		this.behaviorFilename = behaviorFilename;
		this.resultsFilename = resultsFilename;
		this.conditionSelection = conditionSelection;
		this.behaviorBlockConditionSelection = behaviorBlockConditionSelection;
		this.numPermutations = numPermutations;
		this.numBootstraps = numBootstraps;
		this.confidenceLevel = confidenceLevel;
	}
	
	public void doTask() throws Exception {
		
		ConcatenateDatamat st = null;
		progress.startTask("","All tasks");
		progress.startTask("Loading input", "Loading input");
//		progress.updateStatus("Loading input", 0);
		// This is messy, should probably do some dynaming class loading here
		switch(imagingType) {
			case 0: case 1:
				switch(plsType) {
					case 0: case 2:
						st = new ConcatenateFmriDatamat(sessionProfiles, conditionSelection);
						break;
					case 1: case 3: case 4:
						st = new ConcatenateFmriBehavioralDatamat(sessionProfiles, conditionSelection, 
								behaviorFilename);
						break;
				}
				break;
			case 2:
				st = new ConcatenatePetDatamat(sessionProfiles, conditionSelection, behaviorFilename);;
				break;
			case 3:
				break;
		}
		// End loading input
		progress.endTask(); // loading input
//		progress.updateStatus("Completed loading input", 1);
		
		//if you delete this if it will work for fMRI also, but it is not tested
		if(imagingType == 2) // if it is PET
		{
			switch(plsType) {
				case 0:
					//for PET deviation code is same with code is same with ComputeBehaviorPlsMain 
					//(necessary check is done by if behav == true which means Behavior or if behav == false  which means Task PLS) 
					new ComputeBehaviorPlsMain(imagingType, st, numBootstraps, numPermutations, 
							confidenceLevel, behaviorFilename, resultsFilename, false, progress);
					break;
				case 1:
					new ComputeBehaviorPlsMain(imagingType, st, numBootstraps, numPermutations, 
							confidenceLevel, behaviorFilename, resultsFilename, true, progress);
					break;
				case 2:
					new ComputeNonRotatedTaskPlsMain(imagingType, st, numBootstraps, numPermutations, 
							contrastFilename, resultsFilename, progress);
					break;
				case 3:
					new ComputeMultiblockPlsMain(imagingType, st, numBootstraps, numPermutations, 
							confidenceLevel, behaviorFilename, resultsFilename, progress);
					break;
				case 4:
					new ComputeNonRotatedBehavPlsMain(st, numBootstraps, numPermutations, confidenceLevel, 
							behaviorFilename, contrastFilename, resultsFilename, progress);
					break;
			}
		}
		else
		{
			//switch for fMRI
			switch(plsType) {
			case 0:
				new ComputeDeviationPlsMain(st, numBootstraps, numPermutations, resultsFilename, progress);
				break;
			case 1:
				new ComputeBehaviorPlsMain(st, numBootstraps, numPermutations, confidenceLevel, 
						behaviorFilename, resultsFilename, progress);
				break;
			case 2:
				new ComputeNonRotatedTaskPlsMain(st, numBootstraps, numPermutations, contrastFilename, 
						resultsFilename, progress);
				break;
			case 3:
				new ComputeMultiblockPlsMain(st, numBootstraps, numPermutations, confidenceLevel, 
						behaviorFilename, resultsFilename, progress);
				break;
			case 4:
				new ComputeNonRotatedBehavPlsMain(st, numBootstraps, numPermutations, confidenceLevel, 
						behaviorFilename, contrastFilename, resultsFilename, progress);
				break;
			}
		}
		progress.endTask(); // all tasks
		progress.complete();
	}
	
	/**
	 * Sets up an analysis given a parameters filename
	 * @param filename
	 * @throws Exception
	 */
	public Analysis(String filename) throws Exception {
		// The filename is a parameters file to load
		
		Map<String, MLArray> paramInfo;
		
		// Open the matfile
		try {
			paramInfo = new NewMatFileReader(filename).getContent();
		} catch (Exception ex) {
			throw new Exception("An error occurred while opening the parameter file " + filename + ".", ex);
		}
		
		MLStructure pls_setup_info = (MLStructure)paramInfo.get("pls_setup_info");
		
		int imagingType = 0;
		
		int plsType = -1;
		if (((MLDouble)pls_setup_info.getField("mean-centering_PLS")).get(0) == 1) {
			plsType = 0;
		} else if (((MLDouble)pls_setup_info.getField("behavior_PLS")).get(0) == 1) {
			plsType = 1;
		} else if (((MLDouble)pls_setup_info.getField("non-rotated_task_PLS")).get(0) == 1) {
			plsType = 2;
		} else if (((MLDouble)pls_setup_info.getField("multiblock_PLS")).get(0) == 1) {
			plsType = 3;
		} else if (((MLDouble)pls_setup_info.getField("non-rotated_behavior_PLS")).get(0) == 1) {
			plsType = 4;
		}
		
		MLStructure session_file_info = (MLStructure)pls_setup_info.getField("session_file_info");
		
		int numGroups = session_file_info.getN();
		Vector<String[]> sessionProfiles = new Vector<String[]>(numGroups);
		
		for (int i = 0; i < numGroups; i++) {
			String[] currSessFiles = MLFuncs.MLCell1dRow2StrArray((MLCell)(session_file_info.
					getField("session_files", i)));
			sessionProfiles.add(i, currSessFiles);
		}
		//Assumption: all session files have the same conditions
		String sessionFileName = sessionProfiles.get(0)[0];
			
		MLStructure sessProfStruct = (MLStructure) new NewMatFileReader(sessionFileName).
				getContent().get("session_info");
		String[] conditions = MLFuncs.MLCell1dRow2StrArray((MLCell)sessProfStruct.getField("condition"));
//			for (int i = 0; i < conditions.length; ++i) {
//				//TODO: read conditionSelection info in from 
//				System.out.println("Reading conditionSelection... ");
//				conditionSelection.add(new
//				}
		
		// Load the data file fields.
		String contrastFilename = ((MLChar) pls_setup_info.getField("contrast_data_filename")).getString(0);

		String behaviorFilename = ((MLChar) pls_setup_info.getField("behavior_data_filename")).getString(0);
		
		// Load the condition selection info.
		MLDouble conditionSelectionInfo = (MLDouble) pls_setup_info.getField("cond_selection");
		int numConds = conditionSelectionInfo.getM();
		int[] conditionSelection = new int[numConds];
		for (int i = 0; i < numConds; i++) {
			conditionSelection[i] = conditionSelectionInfo.get(i).intValue();
		}
		
		// Load the results file name.
		String resultsFilename = ((MLChar) pls_setup_info.getField("results_filename")).getString(0);
		
		// Load the behavior block condition selection info.
		MLDouble behavBlockConditionSelectionInfo = (MLDouble) pls_setup_info.getField(
				"behav_block_cond_selection");
		numConds = behavBlockConditionSelectionInfo.getM();
		int[] behaviorBlockConditionSelection = new int[numConds];
		for (int i = 0; i < numConds; i++) {
			behaviorBlockConditionSelection[i] = behavBlockConditionSelectionInfo.get(i).intValue();
		}
		
		String numPermutations = ((MLChar) pls_setup_info.getField("num_permutations")).getString(0);

		String numBootstraps = ((MLChar) pls_setup_info.getField("num_bootstraps")).getString(0);

		String confidenceLevel = ((MLChar) pls_setup_info.getField("confidence_level")).getString(0);
		
		this.imagingType = imagingType;
		this.plsType = plsType;
		this.sessionProfiles = sessionProfiles;
		this.contrastFilename = contrastFilename;
		this.behaviorFilename = behaviorFilename;
		this.resultsFilename = resultsFilename;
		this.conditionSelection = conditionSelection;
		this.behaviorBlockConditionSelection = behaviorBlockConditionSelection;
		this.numPermutations = Integer.parseInt(numPermutations);
		this.numBootstraps = Integer.parseInt(numBootstraps);
		this.confidenceLevel = Double.parseDouble(confidenceLevel);
	}
}
