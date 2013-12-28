package pls.analysis;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;

import com.jmatio.types.MLArray;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLStructure;

import extern.NewMatFileReader;
import pls.chrome.sessionprofile.SessionProfileFrame;
import pls.chrome.shared.FilePathCheck;
import pls.sessionprofile.RunInformation;
import pls.shared.GlobalVariablesFunctions;
import pls.shared.MLFuncs;
import npairs.NpairsjSetupParams;
import npairs.io.NpairsDataLoader;

/** NpairsResultSaverPrep does not actually concatenate any datamats, since NPAIRS does not 
 *  require a datamat to be created before analysis is run.  It just prepares a 'ConcatenateDatamat'
 *  object to be used in ResultSaver.addStObjects(ConcatenateDatamat) when adding pertinent info
 *  to results .mat file in the same manner as for PLS .mat results files.
 *  @author anita 
*/

public class NpairsResultSaverPrep extends ConcatenateDatamat {
	
	
	public NpairsResultSaverPrep(Vector<String[]> sessionProfiles, int[] conditionSelection,
			NpairsjSetupParams nsp, NpairsDataLoader npairsDataLoader) throws Exception {
		super(sessionProfiles, conditionSelection, nsp, npairsDataLoader);
		
	}
	
	// for NPAIRS analysis, stackDatamat() does nothing but assign values to variables 
	// ConcatenateDatamat.subjectName, ConcatenateDatamat.eventList
	public void stackDatamat() throws Exception {
		
		int count = 0;
		
		// Go through each subject which is represented by each profile
		for(int i = 0; i < numGroups; i++) {			
			int[] groupTempNewEventList = null;
			
			for(int j = 0; j < sessionGroup[i]; j++, count++) {
				int numRuns = getNumRuns(count);
								
				// get subj_name 
				int[] thisSubjOrder = new int[numRuns];
				
				int firstCond = 0;
				
				while(firstCond < numRuns) {
					thisSubjOrder[firstCond] = 1;
					firstCond += numConditions;			
					subjectName.add("Subj" + (count + 1));
				}
	
				int[] tempEventList = getEventList(j);
				groupTempNewEventList = MLFuncs.append(groupTempNewEventList, tempEventList);
			}
			int[] sortedIndex = MLFuncs.getSortedIndex(groupTempNewEventList);
			int[] groupTempNewEventList3 = MLFuncs.getItemsAtIndices(groupTempNewEventList, sortedIndex);
			
			eventList = MLFuncs.append(eventList, groupTempNewEventList3);
		}
	}
	
	protected void getSessionInfo(String fileSuffix) throws Exception {
		if (npairsDataLoader.loadDatamats()) {
//			System.out.println("Getting session info from super class...");
			// get info from datamats
			super.getSessionInfo(fileSuffix);
			// each time lag (window) gets its own condition (class) label in 
			// an npairs datamat analysis
			String[] npairsEvtRelConds = new String[conditions.length * winSize];
			int[] npairsEvtRelCondSel = new int[conditions.length * winSize];
			for (int c = 0; c < conditions.length; ++c) {
				for (int w = 0; w < winSize; ++w) {
					int currLag = w + 1;
					npairsEvtRelConds[(c * winSize) + w] = conditions[c] + "-" + currLag;
					npairsEvtRelCondSel[(c * winSize) + w] = conditionSelection[c];
				}
			}
			conditions = npairsEvtRelConds;
//			System.out.println("Npairs event-rel conditions: ");
//			NpairsjIO.print(conditions);
			conditionSelection = npairsEvtRelCondSel;	
			
			winSize = 1; // set to 1 so that results viewer will not try to interpret output data
			             // as including winSize > 1 no. of images in each dim.
			
		}
		else { // get info directly from data files

			subjectGroup = new int[numGroups];
			int count = 0;
			int numProfilesLoaded = 0;
			String[] prevConditions = null;
			FilePathCheck check = new FilePathCheck();
			for(int i = 0; i < numGroups; i++) {

				boolean dataLoaded = false;
				int numSessions = sessionGroup[i];
				for(int j = 0; j < numSessions; j++) {
					String sessionFile = check.getExistingFilePath(SESSION, profileList.get(count), currentDirectory);
					if (sessionFile == null) {
						profileList.remove(count);
						sessionGroup[i]--;
						continue;
					}

					Map<String, MLArray> sessionMap = null;
					MLStructure sessionInfo = null;
					try {
						sessionMap = new NewMatFileReader(sessionFile).getContent();
						sessionInfo = (MLStructure) new NewMatFileReader(sessionFile).getContent().get("session_info");
					} catch(Exception e) {
						GlobalVariablesFunctions.showErrorMessage("Session information could not be read from file " + sessionFile + ".");
						profileList.remove(count);
						sessionGroup[i]--;
						continue;
					}

					MLChar createVer = (MLChar)sessionMap.get("create_ver");

					if (!npairsjSetupParams.useCondsAsClasses) {
						int[] uniqConds = MLFuncs.sortAscending(MLFuncs.unique(npairsjSetupParams.getClassLabels()));
						conditions = new String[uniqConds.length];
						for (int c = 0; c < uniqConds.length; ++c) {
							conditions[c] = Integer.toString(uniqConds[c]);
						}
					}
					else {
						conditions = MLFuncs.MLCell1dRow2StrArray((MLCell)sessionInfo.getField("condition0"));
					}

					HashMap<String, MLArray> currInfo = new HashMap<String, MLArray>();
					currInfo.put("sessionFile", new MLChar("sessionFile", sessionFile));
					currInfo.put("createVer", createVer);
					currInfo.put("datamatFilename", new MLChar("st_datamatFile", "")); // no datamat in NPAIRS but required 
					// field for PLS compatibility
					currInfo.put("num_conditions", sessionInfo.getField("num_conditions"));
					currInfo.put("num_runs", sessionInfo.getField("num_runs"));

					//boolean isBlocked = !this.npairsjSetupParams.eventRelAnalysis;
					boolean blockSessFiles = this.npairsjSetupParams.blockSessFiles;
					
					Vector<RunInformation> runInfo = 
						SessionProfileFrame.getRunInformation((MLStructure)sessionInfo.getField("run"), blockSessFiles);

					String dataFileName = runInfo.get(0).dataFiles.split("\\s")[0];
					String dataDir = runInfo.get(0).dataDirectory;
					
					origin = this.npairsDataLoader.getOrigin();
					int[] dims3d = this.npairsDataLoader.getDims();
					dims = new int[] { dims3d[0], dims3d[1], 1, dims3d[2] }; // t comes before z!
				//	float[] voxSizeFloat = this.npairsDataLoader.getVoxSize();
					voxelSize = this.npairsDataLoader.getVoxSize();
						//new int[] { (int)voxSizeFloat[0], (int)voxSizeFloat[1], (int)voxSizeFloat[2] };
					winSize = 1;
					coords = this.npairsDataLoader.getMaskCoords();
					
					int[] eventList = new int[conditions.length];
					for (int k = 0; k < eventList.length; ++k) {
						eventList[k] = k + 1;
					}

					currInfo.put("st_coords", new MLDouble("st_coords", MLFuncs.toDoubleArray(MLFuncs.plus(coords, 1))));
					currInfo.put("st_dims", new MLDouble("st_dims", MLFuncs.toDoubleArray(dims)));
					currInfo.put("st_win_size", new MLDouble("st_win_size", new double[][]{{winSize}}));
					currInfo.put("st_voxel_size", new MLDouble("st_voxel_size", MLFuncs.to2DArray(voxelSize)));
					currInfo.put("st_origin", new MLDouble("st_origin", MLFuncs.toDoubleArray(origin)));
					currInfo.put("st_evt_list", new MLDouble("st_evt_list", MLFuncs.toDoubleArray(eventList)));

					// don't record behavdata or behavname info since these things don't exist 
					// when doing NPAIRS analysis

					dataLoaded = true;
					stInfo.add(currInfo);

					subjectGroup[i] += 1; 

					// Make sure the input data from all sessionfiles is compatible
					if(count > 0) {
						if(!Arrays.equals(getDims(count), getDims(count - 1))) {
							throw new Exception("Sessionfiles have different input data volume dimensions.");
						}
						if(getWindowSize(count) != getWindowSize(count - 1)) {
							throw new Exception("Sessionfiles have different window sizes.");
						}
						if(!Arrays.equals(getVoxelSize(count), getVoxelSize(count - 1))) {
							throw new Exception("Sessionfiles have different input data voxel sizes.");
						}
						if(!Arrays.equals(conditions, prevConditions)) {
							throw new Exception("Sessionfiles do not specify same conditions.");
						}
					}
					prevConditions = conditions;
					count++;
				}
				numProfilesLoaded += sessionGroup[i];
				if (!dataLoaded) {
					throw new Exception("No sessionfiles could be loaded for Group " + (i + 1));
				}
			}
			numConditions = conditions.length;
			numProfiles = numProfilesLoaded;
		}
	}
}
