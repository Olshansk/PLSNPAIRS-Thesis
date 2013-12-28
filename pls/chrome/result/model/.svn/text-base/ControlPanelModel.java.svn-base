package pls.chrome.result.model;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

import pls.chrome.result.controller.observer.filters.IncorrectLagsSelectedEvent;
import pls.chrome.result.controller.observer.filters.SliceFiltersEvent;

public class ControlPanelModel {
	//Associative map between a result file and the lags for which we want to
	//display for that file.
	private HashMap<String, ArrayList<ArrayList<Integer>>> mLagNums =
			new HashMap<String, ArrayList<ArrayList<Integer>>>();

	// A list of the viewed lag numbers for each view (Axial, Sagittal, Coronal)
	//This represents the currently manually selected set of lags that the user
	//desires to view exclusively.
	private ArrayList<ArrayList<Integer>> mLagNumbers = new ArrayList<ArrayList<Integer>>();
	
	// A list of the viewed slices for each view (Axial, Sagittal, Coronal)
	private ArrayList<ArrayList<Integer>> mSliceNumbers = new ArrayList<ArrayList<Integer>>();
	
	// A list of the number of rows per lag for each view (Axial, Sagittal, Coronal)
	private ArrayList<Integer> mNumRowsPerLag = new ArrayList<Integer>();

	//Variables for holding the dimensions of the last known loaded result file.
	private int[] pDimensions = new int[3]; //{axial,sagittal,coronal} sizes

	private GeneralRepository mRepository = null;
	
	//Flag indicating whether the display all lags checkbox is checked or not.
	//{axial,sagittal,coronal} sizes
	private boolean[] displayAllLags = {true,true,true};
	
	public ControlPanelModel(GeneralRepository repository) {
		mRepository = repository;
	}
	
	public ArrayList<Integer> getSliceNumbers(int brainView) {
		return mSliceNumbers.get(brainView);
	}
	
	public void setSliceNumbers(int brainView, ArrayList<Integer> sliceNumbers) {
		mSliceNumbers.set(brainView, sliceNumbers);
	}
	
	public ArrayList<Integer> getLagNumbers(int brainView) {
		return mLagNumbers.get(brainView);
	}

	/**
	 * Saves a record of the selected lags from the user, then modifies the
	 * viewable set of lags.
	 * @param brainView The brainview we are looking at.
	 * @param lagNumbers The set of lags desired to be viewed.
	 */
	public void setLagNumbers(int brainView, ArrayList<Integer> lagNumbers) {
		//Save a record of the lags the user has manually selected.
		mLagNumbers.set(brainView, lagNumbers);
		//Now update the set of lags which this program will display.
		setSpecificLags(brainView, lagNumbers);
	}

	/**
	 * Returns an arraylist containing the set of lags that are currently being
	 * displayed for the requested file.
	 * @param file The result file we wish to focus on.
	 * @param brainview The brain view we are looking at.
	 * @return See description.
	 */
	public ArrayList<Integer> getViewableLags(String file, int brainview){
		return mLagNums.get(file).get(brainview);
	}

	public int getNumRowsPerLag(int brainView) {
		return mNumRowsPerLag.get(brainView);
	}
	
	private void setNumRowsPerLag(int brainView, int numRowsPerLag) {
		mNumRowsPerLag.set(brainView, numRowsPerLag);
	}

	/**
	 *
	 * @param brainView
	 * @return Returns whether the "display all lags" checkbox is selected
	 * for the particular brainview.
	 */
	public boolean getAllLagsFlag(int brainView){
		return displayAllLags[brainView];
	}

	/**
	 * Used whenever a slice filter event occurs.
	 * @param brainView
	 * @param value Sets the truth value of whether or not the "display all
	 * lags" checkbox is selected to value.
	 */
	public void setAllLagsFlag(int brainView, boolean value){
		displayAllLags[brainView] = value;
	}

	/**
	 * This function checks what slices should be displayed when a new file
	 * is added. There are two cases. In the first case the added file is the
	 * same dimensions as the current set of loaded result files (or this is
	 * a new file but we previously just had result files that were of the same
	 * dimension) then we do not need to change which slices are currently
	 * selected to be viewed. In the second case a file with a new dimension
	 * has been loaded and we default to viewing all the slices that this file
	 * has to offer (resetting slices chosen in the control pannel).
	 */
	public void checkSliceNumbers(){

		ResultModel model = mRepository.getGeneral();
		boolean reset = false;

		for(int bv = 0; bv < 3; bv++){
			if (pDimensions[bv] != model.getBrainData().getNumSlices(bv)){
					reset = true;
					break;
			}
		}

		if(reset) //The dimensions were not the same so reset.
			initSliceNumbers();
	}

	/**
	 * This function removes information relating to which lags to display
	 * for a given result file if this result file no longer exists.
	 */
	public void removeLagDisplayInfo(){
		ArrayList<String> filesToRemove = new ArrayList<String>();
		Set<String> resultModels = mRepository.getModels();
		
		for (String models : mLagNums.keySet()){
			if(!resultModels.contains(models))
				filesToRemove.add(models);
		}
		
		for (String modelToRemove : filesToRemove)
			mLagNums.remove(modelToRemove);
	}
	
	/**
	 * Filtering lags for Npairs files does not make sense since Npairs files
	 * always have only a single lag, just record this Npairs model as 
	 * displaying its only lag. 
	 * @param model The npairs model to add selected lag information for.
	 */
	private void setNpairsLags(String model){
		if(mLagNums.containsKey(model)){
			return;
		}
		
		ArrayList<ArrayList<Integer>> displayLags;
		ArrayList<Integer> bv;
		displayLags = new ArrayList<ArrayList<Integer>>(3);
		
		for(int i = 0; i < 3; i++){
			bv = new ArrayList<Integer>(1);
			bv.add(0);
			displayLags.add(bv);
		}
		
		mLagNums.put(model, displayLags);
	}
	
	/**
	 * This function is called every time a file is removed or added.
	 * This function does nothing in the event of removed files but when new
	 * files are added this function will select which lags to display for that
	 * file based on the selection critera the user has provided. This function
	 * will generate a warning if any of the lags the user has attempted to
	 * select are not applicable for the loaded file.
	 */
	public void checkLagNumbers() {
		//Do we need to warn the user that a manually selected lag does not
		//exist for at least one loaded result file?
		boolean warn = false;
		HashMap<String, ArrayList<ArrayList<Integer>>> warnings =
				new HashMap<String, ArrayList<ArrayList<Integer>>>();

		for (String model : mRepository.getModels()){
			
			//Npairs file, skip lag filtering.
			if(mRepository.getGeneral(model) instanceof NPairsResultModel){
				setNpairsLags(model);
				continue;
			}
			
			int numLags = mRepository.getGeneral(model).getWindowSize();

			//If we have a record of which lags to display for this file
			//continue to the next file.
			if(mLagNums.containsKey(model))
				continue;

			ArrayList<ArrayList<Integer>> bvLags = new
					ArrayList<ArrayList<Integer>>();
			ArrayList<ArrayList<Integer>> bvLagsInvalid = new
					ArrayList<ArrayList<Integer>>();

			//For each brain view check if we want to display all the lags or
			//apply manual selection.
			for(int bv=0; bv<3; bv++){
				ArrayList<Integer> lags = new ArrayList<Integer>();
				ArrayList<Integer> invalidLags = new ArrayList<Integer>();
				
				if (displayAllLags[bv]) {
					//Display all lags for each brainview.
					for (int i = 0; i < numLags; i++) {
						lags.add(i);
					}
				}
				else { //display only select lags
					ArrayList<Integer>lagsPerBV = mLagNumbers.get(bv);

					for(Integer lag : lagsPerBV){
						if(lag >= numLags){ //asking for a lag that DNE.
							invalidLags.add(new Integer(lag.intValue()));
							warn = true;
						}else{
							lags.add(new Integer(lag.intValue()));
						}
					}
				}
				bvLags.add(lags);
				bvLagsInvalid.add(invalidLags);
			}
			//save which lags to view for which result file
			mLagNums.put(model, bvLags);

			//At least one view had lags that were non selectable. warn the usr.
			if(warn == true)
				warnings.put(model,bvLagsInvalid);
			warn = false;
		}

		//We have warnings to report
		if(warnings.size() > 0){
			mRepository.getPublisher().publishEvent(
					new IncorrectLagsSelectedEvent(warnings));
		}
	}

	public ArrayList<Integer> initLagNumbers() {
		//windowSize is a PLS term for number of lags.
		int windowSize = mRepository.largestWindowSize();
		
		// Creates three sets of lag numbers, with each set representing a
		// single brain view's visible lags. The default is having each set
		// contain all the lag numbers, which is from 0 to the number of
		// lags - 1, since the lag values are 0-based still.
		ArrayList<Integer> lags = null;
		for (int i = 0; i < 3; ++i) {
			lags = new ArrayList<Integer>();
			for (int j = 0; j < windowSize; ++j) {
				lags.add(j);
			}
			mLagNumbers.add(lags);
		}
		return lags;
	}
	
	@SuppressWarnings("static-access")
	/**
	 * When a file is loaded the slice numbers for the three axis are set
	 * here.
	 */
	public void initSliceNumbers() {
		ResultModel model = mRepository.getGeneral();
		mSliceNumbers = new ArrayList<ArrayList<Integer>>();
		// Creates three sets of slice numbers, with each set representing a
		// single brain view's visible slices. The default is having each set
		// contain all the slice numbers, which is from 1 to the number of
		// slices.
		for (int i = 0; i < 3; ++i) {
			int numSlices = model.getBrainData().getNumSlices(i);
			
			ArrayList<Integer> slices = new ArrayList<Integer>();
			for (int j = 1; j <= numSlices; ++j) {
				slices.add(j);
			}
			mSliceNumbers.add(slices);
		}
		
		//Record the dimensions of the loaded result file.
		pDimensions[0] = mSliceNumbers.get(model.getBrainData().AXIAL).size();
		pDimensions[1] = mSliceNumbers.get(model.getBrainData().SAGITTAL).size();
		pDimensions[2] = mSliceNumbers.get(model.getBrainData().CORONAL).size();
	}
	
	private void initNumRowsPerLag() {
		
		// By default, one row per lag
		for (int i = 0; i < 3; ++i) {
			mNumRowsPerLag.add(1);
		}
	}
	
	/**
	 * Set all lags visible for each file for the specified brainView
	 * @param brainView the brain view.
	 */
	private void displayAllLags(int brainView){
		Set<String> models = mLagNums.keySet();

		for (String model : models){
			int numLags = mRepository.getGeneral(model).getWindowSize();
			ArrayList<Integer> lags = new ArrayList<Integer>();

			for(int i = 0; i< numLags; i++){
				lags.add(new Integer(i));
			}
			mLagNums.get(model).set(brainView, lags);
		}
	}

	/**
	 * Set only the specified lags visible. Some lags may not be applicable
	 * to certain result files and the user will be warned of this.
	 * @param brainView The brainview
	 * @param lagNumbers The particular lags the user wants to view exlusively.
	 */
	private void setSpecificLags(int brainView, ArrayList<Integer> lagNumbers){

		HashMap<String, ArrayList<ArrayList<Integer>>> warnings =
				new HashMap<String, ArrayList<ArrayList<Integer>>>();

		boolean warn = false;

		Set<String> models = mLagNums.keySet();

		//For each model determine which are the valid and invalid lags.
		//Display only the valid lags.
		for (String model : models){
			
			//Do not filter npairs files.
			if( mRepository.getGeneral(model) instanceof NPairsResultModel){
				continue;
			}
			
			ArrayList<Integer> validLags = new ArrayList<Integer>();
			ArrayList<Integer> invalidLags = new ArrayList<Integer>();

			int numLags = mRepository.getGeneral(model).getWindowSize();
			for (int n : lagNumbers){
				if(n >= numLags){
					invalidLags.add(new Integer(n));
					warn = true;
				}else{
					validLags.add(new Integer(n));
				}
			}
			mLagNums.get(model).set(brainView, validLags);

			//insert the invalid lags into their proper brainview
			if(invalidLags.size() > 0){
				ArrayList<ArrayList<Integer>> bvWarnings = new
						ArrayList<ArrayList<Integer>>();
				for(int bv = 0; bv < 3; bv++){
					if(bv == brainView){
						bvWarnings.add(invalidLags);
					}else{
						//for the other two brainviews we aren't looking at
						//create an empty list of invalid lags.
						bvWarnings.add(new ArrayList<Integer>());
					}
				}
				warnings.put(model, bvWarnings);
			}
		}

		//We have warnings to report
		if(warnings.size() > 0){
			mRepository.getPublisher().publishEvent(
					new IncorrectLagsSelectedEvent(warnings));
		}

	}
	
	public void setFilters(int brainView, ArrayList<Integer> lagNumbers,
			ArrayList<Integer> sliceNumbers, int numRowsPerLag, boolean displayAll) {

		//Do not modify the manually selected lags set. We will want to return
		//to that set if we uncheck "display all".
		if(displayAll){
			displayAllLags(brainView);
			setAllLagsFlag(brainView,true);
		}else{
			setLagNumbers(brainView, lagNumbers);
			setAllLagsFlag(brainView,false);
		}
		
		setSliceNumbers(brainView, sliceNumbers);
		setNumRowsPerLag(brainView, numRowsPerLag);
		
		mRepository.getPublisher().publishEvent(new SliceFiltersEvent() );
	}

	public void initModel() {
		initLagNumbers();
		checkLagNumbers();
		initSliceNumbers();
		initNumRowsPerLag();
	}
}