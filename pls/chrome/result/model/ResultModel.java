package pls.chrome.result.model;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;
import java.util.TreeSet;

import pls.chrome.result.controller.Publisher;
import pls.chrome.result.controller.observer.datachange.FlipVolumeEvent;
import pls.chrome.result.controller.observer.selection.SelectedDataTypeChangedEvent;
import pls.chrome.result.controller.observer.selection.SelectedLvChangedEvent;

/**
 * A repository for information retrieved from a results file.
 */
public abstract class ResultModel {
	private ArrayList<ArrayList<String>> mSessionProfiles = null;
	private ArrayList<ArrayList<String>> mSubjectNames = null;
	private ArrayList<String> mConditionNames = null;
	private int[] mConditionSelection = null;
	private int[] mNumSubjectList = null;
	private String mContrastFilename = null;
	private double[][] mS = null;
	private double[][] mDesignLv = null;
	private double[][] mDesignScores = null;
	private double[][] mBrainScores = null;
	private Integer mWindowSize = null;
	private HashMap<String, BrainData> mBrainDataMap = new HashMap<String, BrainData>();
	private BrainData mCurrBrainData = null;
	private String mFilename = null;
	private String mFileDir = null;
	private int[] mCoordinates = null;
	private TreeSet<Integer> mCoordinatesSet = null;
	
	private BrainFilter mBrainFilter = null;
	
	private int[] mDimensions = null;
	private int[] mOrigin = null;
	private double[] mVoxelSize = null;
	private String mCurrDatatype = "";
	private Publisher mPublisher = null;
	private SelectionModel mSelectionModel = null;
	private ViewModel mViewModel = null;
	private ArrayList<String> mMirrorHistory = new ArrayList<String>();
	private double[] mBgImageData = null;
	private ArrayList<String> dataNames = new ArrayList<String>(); //
	
	public ResultModel() {
		mSelectionModel = new SelectionModel();
		mViewModel = new ViewModel();
	}
	
	public void setBrainFilter(BrainFilter filter) {
		mBrainFilter = filter;
		
		for (BrainData bData : mBrainDataMap.values() ) {
			// tells the brain slices to be recreated
			bData.setBrainFilter(filter);
			bData.makeBrainSlices();
		}
	}
	
	// Session Profiles
	public void setSessionProfiles(ArrayList<ArrayList<String>> sessionProfiles) {
		mSessionProfiles = sessionProfiles;
	}
	
	public ArrayList<ArrayList<String>> getSessionProfiles() {
		return mSessionProfiles;
	}
	
	public ArrayList<String[]> getSessionProfileArray() {
		ArrayList<String[]> sessionProfilesArray = new ArrayList<String[]>();
		
		for (int i = 0; i != mSessionProfiles.size(); i++) {
			ArrayList<String> currGroupProfiles = mSessionProfiles.get(i);
			String[] currGroupProfilesArray = new String[currGroupProfiles.size()];
			
			currGroupProfiles.toArray(currGroupProfilesArray);
			sessionProfilesArray.add(currGroupProfilesArray);
		}
		
		return sessionProfilesArray;
	}
	
	// Subject Names
	public void setSubjectNames(ArrayList<ArrayList<String>> subjectNames) {
		mSubjectNames = subjectNames;
	}
	
	public ArrayList<ArrayList<String>> getSubjectNames() {
		return mSubjectNames;
	}
	
	// Condition Names
	public void setConditionNames(ArrayList<String> conditionNames) {
		mConditionNames = conditionNames;
	}
	public ArrayList<String> getConditionNames() {
		return mConditionNames;
	}
	
	// Condition Selection
	public void setConditionSelection(int[] conditionSelection) {
		mConditionSelection = conditionSelection;
	}
	
	public int[] getConditionSelection() {
		return mConditionSelection;
	}
	
	// Num Subject List (number of subjects per group)
	public void setNumSubjectList(int[] numSubjectList) {
		mNumSubjectList = numSubjectList;
	}
	
	public int[] getNumSubjectList() {
		return mNumSubjectList;
	}
	
	// Contrast Filename
	public void setContrastFilename(String contrastFilename) {
		mContrastFilename = contrastFilename;
	}
	
	public String getConstrastFilename() {
		return mContrastFilename;
	}
	
	// S
	public void setS(double[][] s) {
		mS = s;
	}
	
	public double[][] getS() {
		return mS;
	}
	
	// Design Lv
	public void setDesignLv(double[][] designLv) {
		mDesignLv = designLv;
	}
	
	public double[][] getDesignLv() {
		return mDesignLv;
	}
	
	// Design Scores
	public void setDesignScores(double[][] designScores) {
		mDesignScores = designScores;
	}
	
	public double[][] getDesignScores() {
		return mDesignScores;
	}
	
	// Brain Scores
	public void setBrainScores(double[][] brainScores) {
		mBrainScores = brainScores;
	}
	
	public double[][] getBrainScores() {
		return mBrainScores;
	}
	
	// Window Size
	public void setWindowSize(int windowSize) {
		mWindowSize = windowSize;
	}
	
	public int getWindowSize() {
		return mWindowSize;
	}
	
	// Origin
	public int[] getOrigin() {
		return mOrigin;
	}

	public void setOrigin(int[] origin) {
		mOrigin = origin;
		
		// Origin information could potentially be out of whack.  Ensure that
		// we do not try to select an invalid voxel.
		mSelectionModel.selectVoxel(Math.max(1, Math.min(origin[0], mDimensions[0]) ),
				Math.max(1, Math.min(origin[1], mDimensions[1]) ),
				1, 0);
	}
	
	// Voxel Size
	public void setVoxelSize(double[] voxelSize) {
		mVoxelSize = voxelSize;
	}
	
	public double[] getVoxelSize() {
		return mVoxelSize;
	}
	
	// Coordinates
	public void setCoordinates(int[] coordinates) {
		mCoordinates = new int[coordinates.length];

		int j = 0;
		mCoordinatesSet = new TreeSet<Integer>();
		for (int i : coordinates) {
			mCoordinatesSet.add(i - 1);
			mCoordinates[j] = i - 1;
			j++;
		}

//		ArrayList<Integer> coords = new ArrayList<Integer>();
//		for (int i : coordinates) {
//			coords.add(i);
//		}
//		mBrainFilter.addFilter("Basic coords", coords);
	}

	// Note: The coordinates being returned here are 1-based due to being
	// originally taken from a Matlab file, so each value in the array
	// will have to be subtracted by 1 before it is used since Java is
	// 0-based.
	public int[] getCoordinates() {
		return mCoordinates;
	}
	
	public TreeSet<Integer> getCoordinatesSet() {
		return mCoordinatesSet;
	}
	
	public TreeSet<Integer> getFilteredCoordinates() {
//		TreeSet<Integer> ret = new TreeSet<Integer>(mCoordinatesSet);
//		ret.retainAll(mBrainFilter.getFilteredCoords() );
		
		return mBrainFilter.getFilteredCoords(mCoordinatesSet);
	}

	// Dimensions
	public void setDimensions(int[] dimensions) {
		mDimensions = dimensions;
	}
	
	public int[] getDimensions() {
		return mDimensions;
	}

	// Filename
	public void setFilename(String filename) {
		mFilename = filename;
	}
	
	public String getFilename() {
		return mFilename;
	}
	
	// Directory
	public void setFileDir(String fileDir) {
		mFileDir = fileDir;
	}
	
	public String getFileDir() {
		return mFileDir;
	}
	
	/**
	 * Used by {NPAIRS,PLS}ResultLoader.java to load bootstrap, average-zscore,
	 * canonical eigen image data and more.
	 * @param dataName The data type e.g Average Z-scored Eigenimage
	 * @param data the data itself
	 */
	public void addBrainData(String dataName, double[][] data) {
		dataNames.add(dataName); //
		mCurrDatatype = dataName; //The type of data we are dealing with

		mCurrBrainData = new BrainData(data, mCoordinatesSet,
					                   mDimensions, mVoxelSize,
									   mOrigin, mWindowSize, dataName);

		mBrainDataMap.put(mCurrDatatype, mCurrBrainData);
		
		ArrayList<Integer> viewedLvs = new ArrayList<Integer>();
		mViewModel.setSelectedDataType(dataName);
		mViewModel.setViewedLvs(viewedLvs);
	}
	
	public ArrayList<String> getDataNames() {
		return dataNames;
	}

	//A Braindata object corresponds to the type of data that it is about
	//i.e key is Average Z-socred eigenimage, value is Braindata object.
	public BrainData getBrainData() {
		return mBrainDataMap.get(mCurrDatatype);
	}
	
	public BrainData getBrainData(String dataName) {
		return mBrainDataMap.get(dataName);
	}

	public void setBrainData(String dataName) {
		mCurrDatatype = dataName;
		mCurrBrainData = mBrainDataMap.get(mCurrDatatype);
		mViewModel.setSelectedDataType(dataName);
		
		mPublisher.publishEvent(new SelectedDataTypeChangedEvent() );
	}
	
	public void setLv(int lvNum) {
		mCurrBrainData.setLv(lvNum);
		mPublisher.publishEvent(new SelectedLvChangedEvent() );
	}
	
	// Selection Model
	public SelectionModel getSelectionModel() {
		return mSelectionModel;
	}
	
	// Publisher
	protected void setPublisher(Publisher publisher) {
		mPublisher = publisher;
		mSelectionModel.setPublisher(publisher);
		mViewModel.setPublisher(publisher);
		
		for (BrainData bData : mBrainDataMap.values() ) {
			bData.setPublisher(publisher);
		}
	}
	
	public Publisher getPublisher() {
		return mPublisher;
	}
	
	//Returns the selectedDataType in the VolumeBrowser tree
        //i.e if npairs, returns "Average Z-Scored Eigenimage" if we selected
        //a cv of this type.
	public String getSelectedDataType() {
		return mCurrDatatype;
	}
	
	// List of all available data types
	public Set<String> getBrainDataTypes() {
		return mBrainDataMap.keySet();
	}

	public ViewModel getViewModel() {
		return mViewModel;
	}
	
	public abstract String getVariableType();
	
	public abstract String getAbbrVariableType();
	
	protected abstract void addToRepository(GeneralRepository repository);
	
	protected abstract void removeFromRepository(GeneralRepository repository);

	public ArrayList<String> getMirrorHistory() {
		return mMirrorHistory;
	}
	
	public void flipVolume(int brainView, boolean horizontal, boolean vertical) {
		if (horizontal) {
			flipHorizontal(brainView);
		}
		if (vertical) {
			flipVertical(brainView);
		}
		
		mPublisher.publishEvent(new FlipVolumeEvent() );
	}

	public void flipHorizontal(int brainView) {
		for (BrainData bData : mBrainDataMap.values() ) {
			bData.flipHorizontal(brainView);
		}
		
		mMirrorHistory.add(BrainData.viewToString(brainView) + "-H");
	}

	public void flipVertical(int brainView) {
		for (BrainData bData : mBrainDataMap.values() ) {
			bData.flipVertical(brainView);
		}
		
		mMirrorHistory.add(BrainData.viewToString(brainView) + "-V");
	}
	
	// Background image data
	public void setBgImageData(double[] bgImageData) {
		mBgImageData = bgImageData;
	}

	public double[] getBgImageData() {
		return mBgImageData;
	}

	public void updatedFilter() {
		for (BrainData bData : mBrainDataMap.values() ) {
			// tells the brain slices to be recreated
			bData.makeBrainSlices();
		}
	}
	
}
