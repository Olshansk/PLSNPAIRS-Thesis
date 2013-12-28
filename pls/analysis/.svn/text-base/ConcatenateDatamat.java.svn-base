package pls.analysis;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;

import javax.swing.JProgressBar;

//import pls.chrome.sessionprofile.SessionProfileFrame;
import pls.chrome.shared.FilePathCheck;
//import pls.sessionprofile.NiftiAnalyzeImage;
//import pls.sessionprofile.RunInformation;
import pls.shared.BfMRIDatamatFileFilter;
import pls.shared.BfMRISessionFileFilter;
import pls.shared.GlobalVariablesFunctions;
import pls.shared.MLFuncs;
import pls.shared.fMRIDatamatFileFilter;

import Jama.Matrix;

import com.jmatio.io.MatFileFilter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLStructure;

//import extern.ArrayFuncs;
import extern.NewMatFileReader;

import npairs.NpairsjSetupParams;
import npairs.io.NpairsDataLoader;
import npairs.io.NpairsjIO;

/**
 * Generic class that compiles all of the necessary session data and stacks the
 * datamat.  Children should explicitly define how to stack their datamats.
 * @author imran
 *
 */
public class ConcatenateDatamat {
	
	static final String SESSION = "session";
	private static final String DATAMAT = "datamat";
	
	public Vector<String[]> sessionProfiles = null;
	//added for PET to store datamat file list
	public Vector<Matrix> datamatList = new Vector<Matrix>();
	public Vector<String[]> datamatProfiles = null;
	Matrix coord_idx=null;
	//public Vector<Integer> num_cond_lst= null;
	public int[] num_cond_lst = null;
	
	public int numGroups = 0;
	
	public int[] sessionGroup = null;
	
	public Vector<String> profileList = new Vector<String>();
	
	public Vector<Matrix> newDataList = new Vector<Matrix>();
	
	public int numProfiles = 0;
	
	public Vector<HashMap<String, MLArray>> stInfo = new Vector<HashMap<String, MLArray>>();
	
	public Matrix posthocData = null;
	
	public Matrix behavData = null;
	
	public int[] conditionSelection = null;
	
	public String[] conditions = null;
	
	public int totalEvents = 0;
	
	public int[] subjectGroup = null;

	public int numConditions = 0;

	public int[] dims = null;
	
	public double[] M = null;
	
	public int numVoxels = 0;
	
	public int winSize = 0;
	
	public double[] voxelSize = null;
	
	public int[] coords = null;
	
	public int[] origin = null;
	
	public int[] eventList = null;
	
	public Vector<int[]> eventListList = new Vector<int[]>();
	
	public Vector<Matrix> behavDataList = new Vector<Matrix>();
	
	public Matrix datamat = null; // The stacked datamat
	
	public int numBehavSubj = 0;
	
	public int[] numSubjectList = null;
	
	public Vector<String> subjectName = new Vector<String>();
	
	public int[] mask = null;
	
	public int eventLength = 0;
	
	public String currentDirectory = ".";
	
	protected boolean isPLS = true; // will be false in case of NPAIRS analysis
//	
	protected NpairsjSetupParams npairsjSetupParams = null;
	
	private JProgressBar progressBar;
	protected NpairsDataLoader npairsDataLoader;
	
	public ConcatenateDatamat(Vector<String[]> sessionProfiles, String currentDirectory, int[] conditionSelection, 
			JProgressBar progress, NpairsjSetupParams nsp, NpairsDataLoader ndl) throws Exception {
		
		String fileSuffix = null;
		if (sessionProfiles.get(0)[0].endsWith(BfMRISessionFileFilter.EXTENSION)) {
			fileSuffix = BfMRIDatamatFileFilter.EXTENSION;
		} 
		else {
			fileSuffix = fMRIDatamatFileFilter.EXTENSION;
		}
		
		this.npairsjSetupParams = nsp;
		this.npairsDataLoader = ndl;
		
		if (npairsjSetupParams != null) {
			isPLS = false;
		}
		
		this.sessionProfiles = sessionProfiles;
		
		this.conditionSelection = conditionSelection;
		
		this.currentDirectory = currentDirectory;
		
		this.progressBar = progress;
		
		getProfileList(sessionProfiles);
		if (progressBar != null) {
			progressBar.setValue(progressBar.getValue() + 1);
		}
		
		getSessionInfo(fileSuffix);
		if (progressBar != null) {
			progressBar.setValue(progressBar.getValue() + 3);
		}
		
		if(conditionSelection == null || conditionSelection.length == 0) {
			this.conditionSelection = MLFuncs.ones(numConditions);
		}
		
		computeCommonCoordinates();
		if (progressBar != null) {
			progressBar.setValue(progressBar.getValue() + 1);
		}
		
		stackDatamat();
		if (progressBar != null) {
			progressBar.setValue(progressBar.getValue() + 3);
		}
		
		if (isPLS) {
			checkValid();
		}
		
		numConditions = (int)MLFuncs.sum(this.conditionSelection);
		conditions = MLFuncs.getItemsAtIndices(conditions, MLFuncs.find(this.conditionSelection, 1));
		if (progressBar != null) {
			progressBar.setValue(progressBar.getValue() + 1);
		}
		
		deselectConditions();
		if (progressBar != null) {
			progressBar.setValue(progressBar.getValue() + 1);
		}
		
	}
	
	public ConcatenateDatamat(Vector<String[]> sessionProfiles, int[] conditionSelection) throws Exception {
		this(sessionProfiles, ".", conditionSelection, null, null, null);
	}
	
	public ConcatenateDatamat(Vector<String[]> sessionProfiles, int[] conditionSelection, NpairsjSetupParams nsp,
			NpairsDataLoader ndl) throws Exception {
		this(sessionProfiles, ".", conditionSelection, null, nsp, ndl);
	}

	/***************Constructor for PET ******************/
	public ConcatenateDatamat(int n, Vector<String[]> datamatProfiles, int[] conditionSelection) throws Exception {
		
		String fileSuffix = "_PETdatamat.mat";
		
		this.datamatProfiles = datamatProfiles;
		
		this.conditionSelection = conditionSelection;
		
		getProfileList(datamatProfiles);
		
		getPetSessionInfo(fileSuffix); 
		
		if(conditionSelection == null || conditionSelection.length == 0) {
			this.conditionSelection = MLFuncs.ones(numConditions);
		}
		computePetCommonCoordinates();
		
		stackDatamat();
		
		//it is done at stackDatamat() function of ConcatenatePetdatamat class
		//this function is overriden at ConcatenatePetdatamat
		//checkValid(); 
		
		numConditions = (int)MLFuncs.sum(this.conditionSelection);
		
		conditions = MLFuncs.getItemsAtIndices(conditions, MLFuncs.find(this.conditionSelection, 1));
		//????? look at this function for Pet is there a need for such a function
		deselectConditions();
	}
	
	private void getPetSessionInfo(String fileSuffix) throws Exception {
				
		subjectGroup = new int[numGroups];
		
		int count = 0;
		String[] prevConditions = null;
		MLStructure sessionInfo = null;
		Map<String, MLArray> datamatMap = null;
		for(int i = 0; i < numGroups; i++) {
			for(int j = 0; j < sessionGroup[i]; j++, count++) {
				System.out.println("numGroups "+numGroups+" sessionGroup "+sessionGroup[i]);
				String datamatFile = profileList.get(count);
				System.out.println("filename"+datamatFile);
				datamatMap = new NewMatFileReader(datamatFile).getContent();
				//sessionInfo = (MLStructure)new NewMatFileReader(datamatFile).getContent().get("session_info");
				
				sessionInfo = (MLStructure)datamatMap.get("session_info");
				
				MLChar createVer = (MLChar)datamatMap.get("create_ver");
				MLChar sessionFile = (MLChar)datamatMap.get("session_file");
				conditions = MLFuncs.MLCell1dRow2StrArray((MLCell)sessionInfo.getField("condition"));
				// Get the datamat filename
				String datamatPrefix = ((MLChar)sessionInfo.getField("datamat_prefix")).getString(0);
				String path = datamatFile.substring(0, datamatFile.indexOf(datamatPrefix));
				String datamatFilename = path + datamatPrefix + fileSuffix;

				//String[] fields = {"sessionFile","num_conditions", "coords", "datamat","dims", "voxel_size", "origin", "behavdata", "behavname"};
				//Map<String, MLArray> datamatInfo = new NewMatFileReader(datamatFilename, new MatFileFilter(fields)).getContent();
				
				HashMap<String, MLArray> currInfo = new HashMap<String, MLArray>();
				
				currInfo.put("behavdata", datamatMap.get("behavdata"));
				currInfo.put("behavname", datamatMap.get("behavname"));
				currInfo.put("coords", datamatMap.get("coords"));
				currInfo.put("createVer", createVer);
				currInfo.put("datamat", datamatMap.get("datamat"));
				currInfo.put("dims", datamatMap.get("dims"));
				currInfo.put("origin", datamatMap.get("origin"));
				currInfo.put("session_file", sessionFile);
				currInfo.put("datamatFilename", new MLChar("datamatFile", datamatFilename));
				currInfo.put("condition", sessionInfo.getField("condition"));
				currInfo.put("num_conditions", sessionInfo.getField("num_conditions"));
				currInfo.put("num_subject", sessionInfo.getField("num_subjects"));
				currInfo.put("subj_name", sessionInfo.getField("subj_name"));
				currInfo.put("voxel_size", datamatMap.get("voxel_size"));

				/*
				currInfo.put("behavdata", datamatInfo.get("behavdata"));
				currInfo.put("behavname", datamatInfo.get("behavname"));
				currInfo.put("coords", datamatInfo.get("coords"));
				currInfo.put("createVer", createVer);
				currInfo.put("datamat", datamatInfo.get("datamat"));
				currInfo.put("dims", datamatInfo.get("dims"));
				currInfo.put("origin", datamatInfo.get("origin"));
				currInfo.put("session_file", sessionFile);
				currInfo.put("datamatFilename", new MLChar("datamatFile", datamatFilename));
				currInfo.put("condition", sessionInfo.getField("condition"));
				currInfo.put("num_conditions", sessionInfo.getField("num_conditions"));
				currInfo.put("num_subject", sessionInfo.getField("num_subjects"));
				currInfo.put("subj_name", sessionInfo.getField("subj_name"));
				currInfo.put("voxel_size", datamatInfo.get("voxel_size"));
				*/
				System.out.println("Curent Info"+i+" "+j+"is founded");
				/*
				 * 
				File dfn = new File(datamatFilename);
				FileDirContext fdc = new FileDirContext(new File (datamatFilename));
				Date creatindate = fdc.etCreationDate();
				currInfo.put("time_stamp",new MLChar("datamatFileTimestamp", datamatFilenameTimestamp));
				*/

				stInfo.add(currInfo);
				subjectGroup[i] = ((MLDouble)stInfo.get(i).get("num_subject")).getReal(0, 0).intValue();
				
				/***********delete this part this is for trace************/
				System.out.println("\nsize stinfo "+stInfo.size());
			//	int [] petcoords = getPetCoords(i);
			//	funcw(petcoords, "coorsdfirstloading"+i+j+".txt");
				
				datamatList.addElement(getDatamat(i));
			//	System.out.println("\ndatamat listeye eklendi "+datamatList.size()+", "+stInfo.size());
				
//				funcw(getDatamat(i), "datamatfirstloading"+i+j+".txt");
				/*************************/
				// Make sure the datamats are compatible
				if(count > 0) {
					if(!Arrays.equals(getPetDims(count), getPetDims(count - 1))) {
						throw new Exception("The datamats have different volume dimension.");
					}
					//if(getWindowSize(count) != getWindowSize(count - 1)) {
					//	throw new Exception("The datamats have different window size.");
					//}
					if(!Arrays.equals(getPetVoxelSize(count), getPetVoxelSize(count - 1))) {
						throw new Exception("The datamats have different voxel size.");
					}
					if(!Arrays.equals(conditions, prevConditions)) {
						throw new Exception("The datamats are created from different conditions.");
					}
				}
				
				prevConditions = conditions;
			}
		}
//		System.out.println("condition length"+conditions.length);
		numConditions = conditions.length;
	}
	public void computePetCommonCoordinates() throws Exception {
		dims = getPetDims(0);
		int size = dims[0] * dims[1] * dims[2] * dims[3];

		// Determine the common coords
		
		M = new double[dims[0] * dims[1] * dims[2] * dims[3]];
		coord_idx = new Matrix(numProfiles, size);
		
		for(int i = 0; i < numProfiles; i++) {
			int[] currCoords = getPetCoords(i); // 1-rel
			for(int j = 0; j < currCoords.length; j++) {
				int coord = currCoords[j]; 
				M[coord - 1]++;
				coord_idx.set(i, currCoords[j], coord_idx.get(i,currCoords[j])+1);
			}
		}
		coords = MLFuncs.find(M, numProfiles);
		coords = MLFuncs.plus(coords, 1); // make it 1-rel again
		
		numVoxels = coords.length;
		
		if(numVoxels == 0) {
			throw new Exception("No common coords among datamats!");
		}
		
		//winSize = getWindowSize(0);
		voxelSize = getPetVoxelSize(0);
		
		origin = getPetOrigin(0);
		
		if(origin == null || origin.equals(new double[]{0, 0, 0})) {
			if(origin == null) {
				origin = new int[]{0, 0, 0};
			}
			origin[0] = (int)Math.floor(dims[0]/2); 
			origin[1] = (int)Math.floor(dims[1]/2); 
			origin[2] = (int)Math.floor(dims[3]/2); 
		}
	}
	
	public int[] getPetOrigin(int sessionNum) {
		return ((MLDouble)stInfo.get(sessionNum).get("origin")).getIntFirstRowOfArray();
	}
	public int[] getPetDims(int sessionNum) {
		return ((MLDouble)stInfo.get(sessionNum).get("dims")).getIntFirstRowOfArray();
	}
	public int[] getPetCoords(int sessionNum) {
		return ((MLDouble)stInfo.get(sessionNum).get("coords")).getIntFirstRowOfArray();
	}
	public double[] getPetVoxelSize(int sessionNum) {
		return ((MLDouble)stInfo.get(sessionNum).get("voxel_size")).getFirstRowOfArray();
	}
	public int getNumSubject(int sessionNum) {
		return ((MLDouble)stInfo.get(sessionNum).get("num_subject")).getReal(0, 0).intValue();
	}
	public int getNumCondition(int sessionNum) {
		return ((MLDouble)stInfo.get(sessionNum).get("num_conditions")).getReal(0, 0).intValue();
	}
	public String getDatamatFileTimeStamp(int sessionNum) {
		return ((MLChar)stInfo.get(sessionNum).get("datamatFilename")).getString(0);
	}
	public MLCell getSubjectName(int sessionNum) {
		return ((MLCell)stInfo.get(sessionNum).get("subj_name"));
	}
	
	/*********************************/
	
	private void getProfileList(Vector<String[]> sessionProfiles) {
		numGroups = sessionProfiles.size();
		sessionGroup = new int[numGroups];
		for(int i = 0; i < numGroups; i++) {
			sessionGroup[i] = sessionProfiles.get(i).length;
			for(int j = 0; j < sessionGroup[i]; j++) {
				profileList.add(sessionProfiles.get(i)[j]);
			}
		}
		numProfiles = profileList.size();
	}

	
	protected void getSessionInfo(String fileSuffix) throws Exception {
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

				conditions = MLFuncs.MLCell1dRow2StrArray((MLCell)sessionInfo.getField("condition"));
				
//				System.out.println("Conditions: ");
//				NpairsjIO.print(conditions);
				String datamatPrefix = "";
				String path = "";
				String datamatFilename = "";
				Map<String, MLArray> datamatInfo = null;
				
				// Get the datamat filename
				datamatPrefix = ((MLChar)sessionInfo.getField("datamat_prefix")).getString(0);
				int index = 0;
				if (sessionFile.contains(File.separator) ) {
					index = sessionFile.lastIndexOf(File.separator);
				}
				path = sessionFile.substring(0, index) + File.separator;
				datamatFilename = path + datamatPrefix + fileSuffix;

				datamatFilename = check.getExistingFilePath(DATAMAT, datamatFilename, currentDirectory);
				if (datamatFilename != null) {
					try {
						String[] fields = {"num_conditions", "st_coords", "st_dims", "st_evt_list", "st_win_size", "st_voxel_size", "st_origin", "behavdata", "behavname"};
						datamatInfo = new NewMatFileReader(datamatFilename, new MatFileFilter(fields)).getContent();
					} catch(Exception e) {
						GlobalVariablesFunctions.showErrorMessage("Datamat information could not be read from file " + datamatFilename + ".");
						profileList.remove(count);
						sessionGroup[i]--;
						continue;
					}
				} else {
					profileList.remove(count);
					sessionGroup[i]--;
					continue;
				}
				
				HashMap<String, MLArray> currInfo = new HashMap<String, MLArray>();
				currInfo.put("sessionFile", new MLChar("sessionFile", sessionFile));
				currInfo.put("createVer", createVer);
				currInfo.put("datamatFilename", new MLChar("st_datamatFile", datamatFilename));
				currInfo.put("num_conditions", sessionInfo.getField("num_conditions"));
				currInfo.put("num_runs", sessionInfo.getField("num_runs"));
				
				currInfo.put("st_coords", datamatInfo.get("st_coords"));
				currInfo.put("st_dims", datamatInfo.get("st_dims"));
				currInfo.put("st_evt_list", datamatInfo.get("st_evt_list"));
				currInfo.put("st_win_size", datamatInfo.get("st_win_size"));
				currInfo.put("st_voxel_size", datamatInfo.get("st_voxel_size"));
				currInfo.put("st_origin", datamatInfo.get("st_origin"));
				currInfo.put("behavdata", datamatInfo.get("behavdata"));
				currInfo.put("behavname", datamatInfo.get("behavname"));
				
				dataLoaded = true;
				
				stInfo.add(currInfo);
				int numEvents;
				
				numEvents = currInfo.get("st_evt_list").getN();
				
				totalEvents += numEvents;
				subjectGroup[i] += numEvents/conditions.length; // i.e. += 1 in NPAIRS case
				
				// Make sure the datamats are compatible
				if(count > 0) {
					if(!MLFuncs.isEqual(getDims(count), getDims(count - 1))) {
						throw new Exception("The datamats have different volume dimension.");
					}
					if(getWindowSize(count) != getWindowSize(count - 1)) {
						throw new Exception("The datamats have different window size.");
					}
					if(!Arrays.equals(getVoxelSize(count), getVoxelSize(count - 1))) {
						throw new Exception("The datamats have different voxel size.");
					}
					if(!MLFuncs.isEqual(conditions, prevConditions)) {
						throw new Exception("The datamats are created from different conditions.");
					}
				}
				
				prevConditions = conditions;
				count++;
			}
			numProfilesLoaded += sessionGroup[i];
			if (!dataLoaded) {
				throw new Exception("No session/datamat files were able to be loaded for Group " + (i + 1));
			}
		}
		numConditions = conditions.length;
		numProfiles = numProfilesLoaded;
		
		winSize = getWindowSize(0); 
	}
	
	private void checkValid() throws Exception {
		System.out.println("base check valid");
		
		int m = datamat.getRowDimension();
		int n = datamat.getColumnDimension();
		
		if(datamat == null || m == 0 || n == 0 || m > n) {
			throw new Exception("Invalid datamat");
		}
	}
	
	public void stackDatamat() throws Exception {
		// The actual stacking should be overridden by the child classes
	}
	
	public void validatePosthocData() throws Exception {
		if(posthocData == null) {
			return;
		}
		
		int numBehavDataCol = 0;
		
		numBehavDataCol = behavData.getColumnDimension();
		
		if(posthocData.getRowDimension() != numBehavDataCol * numConditions * numGroups) {
			throw new Exception("Rows in Posthoc data file do not match");
		}
	}
	
	public void deselectConditions() {
		MaskEventList fmel = new MaskEventList(eventList, conditionSelection);
		
		eventList = fmel.eventList;
		mask = fmel.mask;
		eventLength = fmel.eventLength;
		
		if (isPLS) {
			datamat = MLFuncs.getRows(datamat, mask);
		}
			numSubjectList = sessionGroup.clone();
		
	}
	
	public void computeCommonCoordinates() throws Exception {
		dims = getDims(0);
		
		// Determine the common coords
		M = new double[dims[0] * dims[1] * dims[2] * dims[3]];
		
		for(int i = 0; i < numProfiles; i++) {
			int[] currCoords = getStCoords(i); // 1-rel coords
			for(int j = 0; j < currCoords.length; j++) {
				int coord = currCoords[j];
				M[coord - 1]++; 
			}
		}
		
		coords = MLFuncs.find(M, numProfiles);
		coords = MLFuncs.plus(coords, 1); // make it 1-rel again
		numVoxels = coords.length;
		
		if(numVoxels == 0) {
			throw new Exception("No common coords among datamats!");
		}
		
//		winSize = getWindowSize(0);
		voxelSize = getVoxelSize(0);
		
		origin = getOrigin(0);
		
		if(origin == null || origin.equals(new double[]{0, 0, 0})) {
			if(origin == null) {
				origin = new int[]{0, 0, 0};
			}
			origin[0] = (int)Math.floor(dims[0]/2); 
			origin[1] = (int)Math.floor(dims[1]/2); 
			origin[2] = (int)Math.floor(dims[3]/2); 
		}
	}
	
	/** Getters for sessionInfo **/
	public Matrix getDatamat(int sessionNum) {
		double [][] val = ((MLDouble)stInfo.get(sessionNum).get("datamat")).getArray();
		//double [][] val = ((MLDouble)stInfo.get(sessionNum).get("datamat")).getArray();
		return new Matrix(val);
	}
	public double getCreateVer(int sessionNum) {
		return new Double(((MLChar)stInfo.get(sessionNum).get("createVer")).getString(0)).doubleValue();
	}
	
	public String getDatamatFilename(int sessionNum) {
		return ((MLChar)stInfo.get(sessionNum).get("datamatFilename")).getString(0);
	}
	
	public int[] getStCoords(int sessionNum) {
		return ((MLDouble)stInfo.get(sessionNum).get("st_coords")).getIntFirstRowOfArray();
	}
	
	public int[] getDims(int sessionNum) {
		return ((MLDouble)stInfo.get(sessionNum).get("st_dims")).getIntFirstRowOfArray();
	}
	
	
	public int[] getEventList(int sessionNum) {
		return ((MLDouble)stInfo.get(sessionNum).get("st_evt_list")).getIntFirstRowOfArray();
	}
	
	public int getNumRuns(int sessionNum) {
		return ((MLDouble)stInfo.get(sessionNum).get("num_runs")).getReal(0, 0).intValue();
	}
	
	public int getWindowSize(int sessionNum) {
		return ((MLDouble)stInfo.get(sessionNum).get("st_win_size")).getReal(0, 0).intValue();
	}
	
	public double[] getVoxelSize(int sessionNum) {
		return ((MLDouble)stInfo.get(sessionNum).get("st_voxel_size")).getFirstRowOfArray();
	}
	
	public int[] getOrigin(int sessionNum) {
		return ((MLDouble)stInfo.get(sessionNum).get("st_origin")).getIntFirstRowOfArray();
	}
	public int[] getBehavData(int sessionNum) {
		MLDouble behavData = (MLDouble)stInfo.get(sessionNum).get("behavdata");
		int[] dims = behavData.getDimensions();
		if(dims[0] == 0 || dims[1] == 0) {
			return new int[]{};
		} else {
			return behavData.getIntFirstRowOfArray();
		}
	}
}