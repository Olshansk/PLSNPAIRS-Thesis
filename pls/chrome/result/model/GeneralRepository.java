package pls.chrome.result.model;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

import javax.swing.JOptionPane;

import pls.chrome.result.ResultLoader;
import pls.chrome.result.ThresholdCalculator;
import pls.chrome.result.blvplot.ColorBarPanel;
import pls.chrome.result.controller.Publisher;
import pls.chrome.result.controller.ResultsCommandManager;
import pls.chrome.result.controller.observer.colourscale.ColourScaleEvent;
import pls.chrome.result.controller.observer.datachange.LoadedVolumesEvent;
import pls.chrome.result.controller.observer.filters.BrainFilterEvent;
import pls.chrome.result.controller.observer.selection.SelectedVolumeChangedEvent;

public class GeneralRepository {
	private TreeMap<String, ResultModel> mResultModels = new TreeMap<String, ResultModel>();
	private TreeMap<String, PlsResultModel> mPlsModels = new TreeMap<String, PlsResultModel>();
	private TreeMap<String, NPairsResultModel> mNPairsModels = new TreeMap<String, NPairsResultModel>();
	private TreeMap<String, ImageOverlayResultModel> mImageOverlayModels = new TreeMap<String, ImageOverlayResultModel>();
	private ControlPanelModel mControlPanelModel = null;
	
	private Publisher mPublisher;// = new Publisher();
	private String mSelectedModel = null;
	
	private File bgImagePath = null;
	private boolean usingNifti = true;
	
	private double[] mCurrentColourScale = null;
	private double[] mCalculatedColourScale = null;
	
	private boolean mUseGlobalScale = true;
	
	private PlotManager mPlotManager;
	private ImagePropertiesModel mImagePropertiesModel;
	
	private BrainFilter mBrainFilter = new BrainFilter();
	                                                 
	private static ColorBarPanel mColorBarPanel = null;
	
	public GeneralRepository() {
		mPublisher = ResultsCommandManager.getPublisher();
		mControlPanelModel = new ControlPanelModel(this);
		mImagePropertiesModel = new ImagePropertiesModel(mPublisher);
	}
	
	public PlotManager getPlotManager() {
		return mPlotManager;
	}
	
	public void setPlotManager(PlotManager plotManager) {
		mPlotManager = plotManager;
	}
	
	public ImagePropertiesModel getImagePropertiesModel() {
		return mImagePropertiesModel;
	}
	
	// Get ControlPanel model
	public ControlPanelModel getControlPanelModel() {
		return mControlPanelModel;
	}
	
	// Returns the result model that matches 'modelName'
	// where 'modelName' is an absolute path i.e /home/.../npairsdata.mat
	public ResultModel getGeneral(String modelName) {
		return mResultModels.get(modelName);
	}

        //Returns the currently selected result model
	public ResultModel getGeneral() {
		return mResultModels.get(mSelectedModel);
	}
	
	// Get pls result model
	public PlsResultModel getPlsModel(String modelName) {
		return mPlsModels.get(modelName);
	}
	
	// Get npairs result model
	public NPairsResultModel getNpairsModel(String modelName) {
		return mNPairsModels.get(modelName);
	}
	
	// Get image overlay result model
	public ImageOverlayResultModel getImageOverlayModel(String modelName) {
		return mImageOverlayModels.get(modelName);
	}
	
	// Selected model
	public void setSelectedModel(String modelName) {
		if (!mSelectedModel.equals(modelName)) {
			mSelectedModel = modelName;
	
			mPublisher.publishEvent(new SelectedVolumeChangedEvent());
		}
	}

	//Set the file path for the currently loaded anatomical image.
	public void setBgImagePath(File path){
		bgImagePath = path;
	}
	
	//Get the file path for the currently loaded anatomical image.
	public File getBgImagePath(){
		return bgImagePath;
	}
	
	public void setAnatomicalLib(boolean useNifti){
		usingNifti = useNifti;
	}
	/**
	 *
	 * @return The anatomical library in use. Since there are only two types
	 * a value of true means that the nifti1 lib is used, false means the
	 * mind seer lib is being used.
	 */
	public boolean getAnatomicalLib(){ return usingNifti; }
	
        //Returns the filename of the selected mat file.
	public String getSelectedResultFile() {
		return mSelectedModel;
	}
	
	// Set of models
	public Set<String> getModels() {
		return mResultModels.keySet();
	}
	
	// Set of pls models
	public Set<String> getPlsModels() {
		return mPlsModels.keySet();
	}
	
	// Set of npairs models
	public Set<String> getNPairsModels() {
		return mNPairsModels.keySet();
	}
	
	// Set of overlay image models
	public Set<String> getImageOverlayModels() {
		return mImageOverlayModels.keySet();
	}

	/**
	 * Adds a new model to the repository. mSelectedModel points to the newly added
	 * model filename.
	 * @param newResultFile filename of the model added.
	 * @param model the pls model to be added.
	 */
	public void addModel(String newResultFile, PlsResultModel model) {
		mResultModels.put(newResultFile, model);
		mPlsModels.put(newResultFile, model);
		model.setBrainFilter(mBrainFilter);
		model.setPublisher(mPublisher);
		
		mSelectedModel = newResultFile;
		mControlPanelModel.checkLagNumbers();
		calculateColourScale();
		mCurrentColourScale = mCalculatedColourScale;
	}
	
	public void addModel(String newResultFile, NPairsResultModel model) {
		mResultModels.put(newResultFile, model);
		mNPairsModels.put(newResultFile, model);
		model.setBrainFilter(mBrainFilter);
		model.setPublisher(mPublisher);
		
		mSelectedModel = newResultFile;
		mControlPanelModel.checkLagNumbers();
		calculateColourScale();
		mCurrentColourScale = mCalculatedColourScale;
	}
	
	public void addModel(String newResultFile, ImageOverlayResultModel model) {
		mResultModels.put(newResultFile, model);
		mImageOverlayModels.put(newResultFile, model);
		model.setBrainFilter(mBrainFilter);
		model.setPublisher(mPublisher);
		
		mSelectedModel = newResultFile;
		//This may be necessary mControlPanelModel.checkLagNumbers();
		//calculateColourScale();
		//mCurrentColourScale = mCalculatedColourScale;
	}
	
	public void removePlsModel(String modelName) {
		mPlsModels.remove(modelName);
		mResultModels.remove(modelName);
	}
	
	public void removeNPairsModel(String modelName) {
		mNPairsModels.remove(modelName);
		mResultModels.remove(modelName);
	}
	
	public void removeImageOverlayModel(String modelName) {
		mImageOverlayModels.remove(modelName);
		mResultModels.remove(modelName);
	}

	/**
	 * Function that is called when user hits 'ok' after selecting result
	 * files to load in the result file browser. This is not the browser
	 * at the start of the program but the custom made result file browser
	 * that is found once the result viewer has already been loaded.
	 * @param volumes set of volumes that are currently active.
	 * @return fails if any of the result files are invalid.
	 */
	public boolean setLoadedFiles(ArrayList<String> volumes) {
		// Figure out what items to remove and get a valid model
		ResultModel validModel = null;
		ArrayList<String> removedItems = new ArrayList<String>();
		double[] bgImage = getGeneral().getBgImageData();

		for (String s : mResultModels.keySet() ) {
			if (!volumes.contains(s) ) {
				removedItems.add(s);
			}
			else if (validModel == null) {
				validModel = mResultModels.get(s);
			}
		}
		
		// Figure out what items to add
		ArrayList<String> addedItems = new ArrayList<String>();
		ArrayList<ResultModel> modelsToAdd = new ArrayList<ResultModel>();
		
		for (String s : volumes) {
			if (!mResultModels.containsKey(s) ) {
								
				ResultLoader loader = ResultLoader.makeLoader(s);
				
				if (loader == null) {
					JOptionPane.showMessageDialog(null, "The file " + s + " does not have a recognized extension.");
					return false;
				}
				
				try {
					loader.loadFile();
				}
				catch (Exception e) {
					JOptionPane.showMessageDialog(null, "An error occurred while opening " + s 
							+ "\n" + e.getMessage());
					return false;
				}
				
				ResultModel model = loader.getResultModel();
								
					if (validModel == null) {
						validModel = model;
					}
					
					if (Arrays.equals(validModel.getDimensions(), model.getDimensions() ) ) {
						addedItems.add(s);
						
						modelsToAdd.add(model);
					}
					else {
						JOptionPane.showMessageDialog(null, 
								"The dimensions of " + s + 
								" are not the same as volumes already loaded.");
						return false;
					}
			}
		}
		
		// If we get here, no errors have occurred so remove the items to be
		// removed and add the items to be added.
		for (String s : removedItems) {
			mResultModels.get(s).removeFromRepository(this);
		}

		mSelectedModel = null;


		//if resultmodels set is empty at this point then we are adding all
		//new files and we want to reset the images. if its not empty
		//then we still have files from a previous run and we want to continue
		//loading the same image. If we are adding all new files, then even if
		//they are of the same dims as the previous run we still want to reset.
		if(mResultModels.isEmpty()){
			bgImage = null;
		}

		//New models that were not in the repository before.
		for (ResultModel model : modelsToAdd) {
			model.addToRepository(this);
			
			String type = model.getSelectedDataType();
			model.getViewModel().getViewedLvs(type).add(1);
			model.setBgImageData(bgImage);
		}

		//This happens if we haven't added any new files.
		if (mSelectedModel == null) {
			mSelectedModel = validModel.getFilename();
		}
		
		mControlPanelModel.removeLagDisplayInfo();
		mControlPanelModel.checkSliceNumbers();

		mPlotManager.refreshPlots();
		mPublisher.publishEvent(new LoadedVolumesEvent());
		
		return true;
	}
	
	public Publisher getPublisher() {
		return mPublisher;
	}

	/**
	 * Sets the global colour scale calculated over all currently viewed LVs.
	 * The following rationale is applied to the threshold: Canonical eigenimages
	 * and BrainLV data are around the same size and Average Z-Scored eigenimages
	 * are around the same size as bootstrap data. Hence it only makes sense
	 * to apply a threshold when we are looking at opened images belonging to
	 * the pairs ({Fulldata/Average} canonical eigenimages, brainlv) and
	 * (average z-scored eigenimages, bootstrap). If we have any other sets
	 * simultaneously open set the threshold to zero. i.e open lvs in (z-score,
	 * brainlv) imples a threshold of 0 is set.
	 *
	 * Post condition: returns min/max over all open lvs. Threshold is 0 if
	 * opened lvs do not belong exclusively to the sets defined above.
	 * Otherwise the threshold is the value calculated by taking the 95th
	 * percentile value of the sorted and aggregated brain values of the open
	 * lvs in (z-score,bootstrap) or is the value 1/3 away from 
	 * (abs(max)+abs(min))/2 when looking at (canonical eigenimages, brainlv).
	 * Min and Max are set to zero if not lvs are open.
	 */
	public void calculateColourScale() {
		double max = Double.MIN_VALUE;
		double min = Double.MAX_VALUE;
		double threshold = 0.0;
		
		ArrayList<Double> data = new ArrayList<Double>();
		
		boolean atLeastOneLvShown = false;
		boolean hasBootstrap = false;
		boolean hasNormal = false;
		boolean hasCanonical = false;
		boolean hasZScore = false;
		boolean compatibleThreshold = true;
		
		for (String resultFile : getModels() ) {
			ResultModel model = mResultModels.get(resultFile);
			
			ViewModel viewModel = model.getViewModel();
			
			for (String dataType : model.getBrainDataTypes() ) {
				ArrayList<HashMap<Integer, double[]>> modelData = model.getBrainData(dataType).getAllData();
				
				ArrayList<Integer> viewedLvs = viewModel.getViewedLvs(dataType);
				
				if (viewedLvs.size() > 0) {
					atLeastOneLvShown = true;
					
					if (dataType.equals(BrainData.BOOTSTRAP_STRING) ) {
						hasBootstrap = true;
					}
					else if (dataType.equals(BrainData.BRAINLV_STRING)) {
						hasNormal = true;
					}
					else if (dataType.equals(BrainData.AVG_ZSCORED_STRING)){
						hasZScore = true;
					}
					else if (dataType.equals(BrainData.AVG_CANONICAL_STRING)
							|| dataType.equals(BrainData.FULL_DATA_STRING)){
						hasCanonical = true;
					}
					else {
						compatibleThreshold = false;
					}

					//invalid mixes
					if ( (hasBootstrap && hasNormal) ||
						 (hasBootstrap && hasCanonical) ||
						 (hasNormal && hasZScore) ||
						 (hasZScore && hasCanonical)) {
						compatibleThreshold = false;
					}

				/*If we have a mix of datatypes we can't have together i.e
				 * Zscore and canonical, don't calculate the threshold but
				 * go ahead and calculate the max and mins nevertheless. */
					for (int lv : viewedLvs) {
						// Get the max and min
						double[] colourScale = model.getBrainData(dataType)
								.getMaxMinThresh(lv - 1);
						
						max = Math.max(max, colourScale[0]);
						min = Math.min(min, colourScale[1]);
						
						HashMap<Integer, double[]> lvData = modelData.get(lv - 1);
						
						if (compatibleThreshold) {
							for (double[] lagData : lvData.values() ) {
								for (double value : lagData) {
									data.add(value);
								}
							}
						}
					}
				}
			}
		}
		
		if (!atLeastOneLvShown) {
			max = 0;
			min = 0;
		}

		//don't calculate the threshold if we have an invalid mix of data.
		if (compatibleThreshold) {
			if (hasBootstrap || hasZScore) {
				threshold = ThresholdCalculator.calculateBootstrapThreshold(data);
			}
			else if (hasNormal || hasCanonical) {
				threshold = ThresholdCalculator.calculateNormalThreshold(max, min);
			}
		}
		
		mCalculatedColourScale = new double[]{max, min, threshold};
		
		setGlobalColourScale(mCalculatedColourScale);
	}
	
	public double[] getCalculatedColourScale() {
		return mCalculatedColourScale;
	}
	
	public double[] getGlobalColourScale() {
		return mCurrentColourScale;
	}
	
	public void setGlobalColourScale(double[] colourScale) {
		mCurrentColourScale = colourScale;
		
		mPublisher.publishEvent(new ColourScaleEvent());
	}

	public void setGlobalColourScale(double max, double min, double threshold) {
		double[] colourScale = new double[]{max, min, threshold};
		
		setGlobalColourScale(colourScale);
	}
	
	public boolean getUseGlobalScale() {
		return mUseGlobalScale;
	}
	
	public void setUseGlobalScale(boolean useScale) {
		mUseGlobalScale = useScale;
		
		mPublisher.publishEvent(new ColourScaleEvent() );
	}

	public void dispose() {
		mPublisher.dispose();
	}
	
	public BrainFilter getBrainFilter() {
		return mBrainFilter;
	}
	
	public void addBrainFilter(String filterName, TreeSet<Integer> filter) {
		mBrainFilter.addFilter(filterName, filter);
		
		for (ResultModel model : mResultModels.values() ) {
			model.updatedFilter();
		}
		
		mPublisher.publishEvent(new BrainFilterEvent() );
	}
	
	public void toggleBrainFilterEnabled(String filterName) {
		mBrainFilter.toggleFilterEnabled(filterName);
		
		for (ResultModel model : mResultModels.values() ) {
			model.updatedFilter();
		}
		
		mPublisher.publishEvent(new BrainFilterEvent() );
	}
	
	public void removeBrainFilter(String filterName) {
		mBrainFilter.removeFilter(filterName);
		
		for (ResultModel model : mResultModels.values() ) {
			model.updatedFilter();
		}
		
		mPublisher.publishEvent(new BrainFilterEvent() );
	}

	public static void setColorBarPanel(ColorBarPanel aColorBarPanel) {
		mColorBarPanel = aColorBarPanel;
	}

	public static ColorBarPanel getColorBarPanel() {
		return mColorBarPanel;
	}
	
	public int largestWindowSize() {
		int maxLags = 0;
		Iterator it = mResultModels.values().iterator();
		
		while(it.hasNext()) {
			int ws = ((ResultModel)it.next()).getWindowSize();
			maxLags = Math.max(ws, maxLags);
		}
		return maxLags;
	}
}
