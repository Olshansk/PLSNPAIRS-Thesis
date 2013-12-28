package pls.chrome.result;

import java.io.File;
import java.util.ArrayList;
import java.util.Map;

import com.jmatio.io.MatFileFilter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLDouble;

import extern.NewMatFileReader;

import pls.chrome.result.model.ResultModel;
import pls.chrome.shared.FilePathCheck;
import pls.shared.AnalyzeImageFileFilter;
import pls.shared.BfMRIResultFileFilter;
import pls.shared.MLFuncs;
import pls.shared.NiftiImageFileFilter;
import pls.shared.NpairsfMRIResultFileFilter;
import pls.shared.fMRIResultFileFilter;

public abstract class ResultLoader {
	public static String NPAIRS_TYPE_STRING = "NPAIRS";
	public static String PLS_TYPE_STRING = "PLS";
	
	private static String[] COMMON_FIELDS = {"SessionProfiles", "ContrastFile", "s",
		"num_conditions", "subj_name", "st_coords", "st_dims", "st_evt_list",
		"st_win_size", "st_voxel_size", "st_origin", "behavdata",
		"behavname", "perm_result", "boot_result", "brainlv",
		"designlv", "b_scores", "d_scores", "cond_name",
		"cond_selection", "num_subj_lst", "brainscores"};
	
	protected ArrayList<String> mRelevantFields = new ArrayList<String>();
	
	protected Map<String, MLArray> mResultInfo = null;
	
	protected String mFilename = null;
	
	protected ResultModel mResultModel = null;
	
	public static ResultLoader makeLoader(String filename) {
		ResultLoader loader = null;
		
		File file = new File(filename);
		
		String shortName = file.getName();
		
		if (shortName.endsWith(NpairsfMRIResultFileFilter.EXTENSION) ) {
			loader = new NpairsResultLoader(filename);
		}
		else if (shortName.endsWith(fMRIResultFileFilter.EXTENSION) ||
				shortName.endsWith(BfMRIResultFileFilter.EXTENSION) ) {
			loader = new PlsResultLoader(filename);
		}
		else if (shortName.endsWith(NiftiImageFileFilter.NIFTI_EXTENSION) ||
				shortName.endsWith(AnalyzeImageFileFilter.IMG_EXTENSION) ||
				shortName.endsWith(AnalyzeImageFileFilter.HDR_EXTENSION)) {
			loader = new ImageOverlayResultLoader(filename);
		}
		
		return loader;
	}
	
	public ResultLoader(String filename) {
		mFilename = filename;
	}
	
	public void loadFile() throws Exception {
		// Add common fields to the collection of fields to grab
		for (String s : COMMON_FIELDS) {
			mRelevantFields.add(s);
		}
		
		// Allow concrete classes to grab fields they need
		addOtherRelevantFields();

		// Open the matfile
		try {
			mResultInfo = new NewMatFileReader(mFilename, new MatFileFilter(
					mRelevantFields.toArray(new String[0]))).getContent();
		} catch (Exception ex) {
			throw new Exception("Result file " + mFilename
					+ " could not be loaded.", ex);
		}
		
		createResultModel(); //mResultModel initialized here.
		
		loadCommonData();
		
		loadOtherData();
	}
	
	public ResultModel getResultModel() {
		return mResultModel;
	}
	
	protected abstract void addOtherRelevantFields();
	
	protected abstract void createResultModel();
	
	private void loadCommonData() {
		mResultModel.setFilename(mFilename);
		mResultModel.setFileDir(new File(mFilename).getParent() );
		
		// Read in brain data.
		
		// Condition Names
		MLCell cond_name = (MLCell) mResultInfo.get("cond_name");
		ArrayList<String> conditionNames = MLFuncs.MLCell1dRow2StrArrayList(cond_name);
		
		mResultModel.setConditionNames(conditionNames);
		
		// Voxel Size
		mResultModel.setVoxelSize(((MLDouble) mResultInfo.get("st_voxel_size")).getFirstRowOfArray());
		
		// Coordinates (st_coords)
		mResultModel.setCoordinates(((MLDouble) mResultInfo.get("st_coords")).getIntFirstRowOfArray());
		
		// Dimensions
		mResultModel.setDimensions(((MLDouble) mResultInfo.get("st_dims")).getIntFirstRowOfArray());
		
		// Origin
		mResultModel.setOrigin(((MLDouble) mResultInfo.get("st_origin")).getIntFirstRowOfArray());
		
		// Window Size (number of lags)
		mResultModel.setWindowSize(((MLDouble) mResultInfo.get("st_win_size")).getReal(0, 0).intValue());
		
		// Condition Selection
		mResultModel.setConditionSelection(((MLDouble) mResultInfo.get("cond_selection")).getIntFirstRowOfArray());
		
		// Num Subject List (number of subjects per group)
		int[] num_subj_lst = ((MLDouble) mResultInfo.get("num_subj_lst")).getIntFirstRowOfArray();
		mResultModel.setNumSubjectList(num_subj_lst);
		
		// DesignLv
		MLArray temp = mResultInfo.get("designlv");
		if (temp != null) {
			mResultModel.setDesignLv(((MLDouble)temp).getArray());
		}

		// Brain Scores
		temp = mResultInfo.get("b_scores"); 
		if (temp != null) {
			mResultModel.setBrainScores(((MLDouble)temp).getArray());
		}
		else {
			temp = mResultInfo.get("brainscores");
			if (temp != null) {
				mResultModel.setBrainScores(((MLDouble)temp).getArray());
			}
		}

		// Design Scores
		temp = mResultInfo.get("d_scores"); 
		if (temp != null) {
			mResultModel.setDesignScores(((MLDouble)temp).getArray());
		}		
		
		// Contrast Filename
		// In matlab, contrastFile stores the design not the filepath.
		
		String contrast = null;
		FilePathCheck check = new FilePathCheck();
		
		if ( mResultInfo.get("ContrastFile").isDouble()) {
			contrast = "NONE";
		}
		else {
		contrast = ((MLChar) mResultInfo.get("ContrastFile")).getString(0);
		
		// If the contrast value is not given as a specific key word, then it
		// is assumed to be a path to an actual contrast text file.
		
		if (!contrast.equals("NONE") && !contrast.equals("BEHAV") && !contrast.equals("MULTIBLOCK")) {
			contrast = check.getExistingFilePath("contrast", contrast, mResultModel.getFileDir() );
			if (contrast == null) {
				contrast = "NONE";
			}
		}
		}
		mResultModel.setContrastFilename(contrast);

		// S
		MLDouble stuff = ((MLDouble) mResultInfo.get("s"));
		if (stuff != null) {
			mResultModel.setS(stuff.getArray());
		}
		
		// Subject Names
		temp = mResultInfo.get("subj_name");
		ArrayList<ArrayList<String>> groupSubjectNames = new ArrayList<ArrayList<String>>();
		if (temp != null && temp.isCell() ) {
			MLCell struct_subj_name = (MLCell) temp;
			
			int k = 0;
			for (int i = 0; i != num_subj_lst.length; i++) {
				ArrayList<String> subjectNames = new ArrayList<String>();
				for (int j = 0; j != num_subj_lst[i]; j++) {
					String subj_name = struct_subj_name.get(k).contentToString();
					int first = subj_name.indexOf('\'') + 1;
					int last = subj_name.lastIndexOf('\'');
					subjectNames.add(subj_name.substring(first, last));
					k++;
				}
				groupSubjectNames.add(subjectNames);
			}
			
			mResultModel.setSubjectNames(groupSubjectNames);
		}
	}
	
	protected abstract void loadOtherData();
}
