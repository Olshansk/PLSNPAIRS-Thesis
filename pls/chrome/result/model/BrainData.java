package pls.chrome.result.model;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.TreeSet;

import org.apache.commons.math.stat.descriptive.rank.Percentile;
import pls.chrome.result.controller.Publisher;
import pls.chrome.result.controller.observer.datachange.InvertedLvEvent;
import pls.chrome.result.controller.observer.selection.SelectedLvChangedEvent;
import pls.shared.GlobalVariablesFunctions;
import pls.shared.MLFuncs;

/**
 * Encapsulates the data of a single window of data
 */
public class BrainData {
	public static final String BRAINLV_STRING = "Brain LV Plot";
	public static final String BOOTSTRAP_STRING = "Bootstrap Ratio Plot";
	
	public static final String FULL_DATA_STRING = "Full-data Reference Canonical Eigenimage";
	public static final String AVG_CANONICAL_STRING = "Average Canonical Eigenimage";
	public static final String AVG_ZSCORED_STRING = "Average Z-Scored Eigenimage";
	
	public static final String IMAGE_OVERLAY_STRING = "Image Overlay";
	
	final static public int AXIAL = 0;
	final static public int SAGITTAL = 1;
	final static public int CORONAL = 2;

	//[lv][1-d coord][lag]
	private ArrayList<HashMap<Integer, double[]> > mData = new ArrayList<HashMap<Integer, double[]> >();
	private int[] mDimensions = null;
	private double[] mVoxelSize = null;
	private int[] mOrigin = null;
	private int mWindowSize = 0;
	
	private ArrayList<ArrayList<Integer>> mAxialSlices = new ArrayList<ArrayList<Integer>>();
	private ArrayList<ArrayList<Integer>> mSagittalSlices = new ArrayList<ArrayList<Integer>>();
	private ArrayList<ArrayList<Integer>> mCoronalSlices = new ArrayList<ArrayList<Integer>>();
	
	// max, min and calculated threshold values
	protected double[] mMaxValues = null;
	protected double[] mMinValues = null;
	protected double[] mThresholds = null;
	
	// The currently selected latent variable
	private int mCurrLv = 0;
	
	// The total number of latent variables
	protected int mNumLvs = 0;
	
	private ColourScaleModel mColourModel;
	private Publisher mPublisher;
	
	private BrainFilter mBrainFilter;
	private TreeSet<Integer> mCoordinates;
	
	//Tells us the type of brain data. Average Z-score, etc.
	private final String identity;

	public BrainData(double[][] data,
			TreeSet<Integer> coordinates,
			int[] dims,
			double[] voxelSize,
			int[] origin,
			int windowSize,
			String type) {
		
		mDimensions = dims;
		mVoxelSize = voxelSize;
		mOrigin = origin;
		mWindowSize = windowSize;
		mCoordinates = coordinates;
		identity = type;

		loadData(data);
		/*when the model this brain data belongs to is added,
		 *makeBrainSlices is called anyways when setBrainFilter is called
		 * see line 135 of GeneralRepository.java so we may not need this call.
		 */
		makeBrainSlices();
		
		mColourModel = new ColourScaleModel(this);
	}
	
	private void loadData(double[][] data) {
		mNumLvs = data[0].length;
		mMaxValues = new double[mNumLvs];
		mMinValues = new double[mNumLvs];
		mThresholds = new double[mNumLvs];
		
		// For each Lv
		for (int i = 0; i < mNumLvs; ++i) {
			mMaxValues[i] = 0.0;
			mMinValues[i] = 0.0;
			//mThresholds[i] = 0.0;
			
			// Shove the data into a hashmap, where the 1-D coordinate is the
			// key and the value is an array containing the values for each lag
			HashMap<Integer, double[]> currVolume = new HashMap<Integer, double[]>();
			int j = 0;
			for (int coord : mCoordinates) {
				double[] d = new double[mWindowSize];
				
				int temporalIndex = j * mWindowSize;

				//For each lag.
				for (int k = 0; k < mWindowSize; ++k) {
					d[k] = data[temporalIndex + k][i];
					if (Double.isNaN(d[k])) {
						String naNMessage = "Brain volume contains NaNs.";
						GlobalVariablesFunctions.showErrorMessage(naNMessage);
						throw new IllegalArgumentException(naNMessage);
					}
					if (d[k] > mMaxValues[i]) {
						mMaxValues[i] = d[k];
					}
					else if (d[k] < mMinValues[i]) {
						mMinValues[i] = d[k];
					}
				}
				
				currVolume.put(coord, d); //map particular coord to data point.
				++j;
			}
			mData.add(currVolume);
		}
		
		calculateThresholds(data);
	}
	
	public void makeBrainSlices() {
		mAxialSlices.clear();
		mSagittalSlices.clear();
		mCoronalSlices.clear();

		//For each slice, add an arrayList to store the coordinates in that
		//slice.

		for (int i = 0; i < getNumSlices(AXIAL); ++i) {
			mAxialSlices.add(new ArrayList<Integer>() );
		}
		
		for (int i = 0; i < getNumSlices(SAGITTAL); ++i) {
			mSagittalSlices.add(new ArrayList<Integer>() );
		}
		
		for (int i = 0; i < getNumSlices(CORONAL); ++i) {
			mCoronalSlices.add(new ArrayList<Integer>() );
		}
		
		if (mBrainFilter != null) {
			TreeSet<Integer> coords = new TreeSet<Integer>(mCoordinates);
			
			for (int coord : mBrainFilter.getFilteredCoords(coords) ) {

				//Translate which slices this 1d coord belongs to for each view.
				//Then add the coord.

				int[] xyslice = convert1DtoView(coord, AXIAL);
				mAxialSlices.get(xyslice[2]).add(coord);
				
				xyslice = convert1DtoView(coord, CORONAL);
				mCoronalSlices.get(xyslice[2]).add(coord);
				
				xyslice = convert1DtoView(coord, SAGITTAL);
				mSagittalSlices.get(xyslice[2]).add(coord);
			}
		}
	}
	
	public double[] getMaxMinThresh() {
		return new double[]{mMaxValues[mCurrLv], mMinValues[mCurrLv], mThresholds[mCurrLv]};
	}
	
	public double[] getMaxMinThresh(int lv) {
		return new double[]{mMaxValues[lv], mMinValues[lv], mThresholds[lv]};
	}
	
	public void setLv(int lv) {
		mCurrLv = lv;
		mColourModel.setCurrLv(lv);
		
		mPublisher.publishEvent(new SelectedLvChangedEvent() );
	}
	
	public int getLv() {
		return mCurrLv;
	}
	
	public int getNumLvs() {
		return mNumLvs;
	}
	
	public int[] getDimensions() {
		return mDimensions;
	}
	
	public double[] getVoxelSize() {
		return mVoxelSize;
	}
	
	public int[] getOrigin() {
		return mOrigin;
	}
	
	/**
	 * Given a 3D coordinate (x, y, z) and lag, returns the
	 * corresponding value from the data volume.
	 */
	public double getValue3D(int x, int y, int z, int lag) {
		return getValue3D(x, y, z, lag, mCurrLv);
	}
	
	/**
	 * Given a 3D coordinate (x, y, z), lag and lv, returns the
	 * corresponding value from the data volume for the given lv.
	 */
	public double getValue3D(int x, int y, int z, int lag, int lv) {
		int index = convert3Dto1D(x, y, z);
		return getValue1D(index, lag);
	}
	
	/**
	 * Given a view, x and y coordinate, slice and lag, returns the
	 * corresponding value from the data volume.
	 */
	public double getValueView(int view, int x, int y, int slice, int lag) {
		return getValueView(view, x, y, slice, lag, mCurrLv);
	}
	
	/**
	 * Given a view, x and y coordinate, slice and lag, returns the
	 * corresponding value from the data volume.
	 */
	public double getValueView(int view, int x, int y, int slice, int lag, int lv) {
		int xAbs = 0, yAbs = 0, zAbs = 0;
		
		switch (view) {
		case AXIAL:
			xAbs = x;
			yAbs = y;
			zAbs = slice;
			break;
		case SAGITTAL:
			xAbs = slice;
			yAbs = x;
			zAbs = y;
			break;
		case CORONAL:
			xAbs = x;
			yAbs = slice;
			zAbs = y;
			break;
		};
		
		return getValue3D(xAbs, yAbs, zAbs, lag, lv);
	}
	
	/**
	 * Checks that the given index is significant (i.e. was contained in
	 * st_coords).
	 * Returns the value for that coordinate if significant,
	 * throws an ArrayIndexOutOfBoundsException otherwise.
	 */
	public double getValue1D(int index, int lag) {
		return getValue1D(index, lag, mCurrLv);
	}
	
	/**
	 * Checks that the given index has an associated brain value (i.e 
	 * contained in st_coords).
	 * Returns the value for that coordinate if available, 0.0 otherwise.
	 * Throws an ArrayOutOfBoundsException if the value does not exist
	 * for the specified lag.
	 * Note: One must be very careful while using this function because it will
	 * ALWAYS return a value unless the coordinate is valid but the desired
	 * lag does not exist. Expect to get back 0 values even for indexs that
	 * are otherwise not contained in st_coords.
	 */
	
	public double getValue1D(int index, int lag, int lv) {

		HashMap<Integer, double[]> volume = mData.get(lv);

		if(!volume.containsKey(index)){
			return 0.0;
		}
		return volume.get(index)[lag]; 
	}

	/**
	 * Does essentially the same job as getValue1D but do not return zero
	 * values if the coordinate does not exist. This function is used for
	 * calculating the correlation value for the scatter plot. If zero values
	 * are returned for indexs that do not exist that the correlation is
	 * incorrectly calculated.
	 * @param index the 1D coordinate
	 * @param lag the lag
	 * @param lv the lv
	 * @return the value at the coordinate,lag,lv.
	 */
	public double getValue1DCorr(int index, int lag, int lv) {

		HashMap<Integer, double[]> volume = mData.get(lv);

		if(!volume.containsKey(index)){
			throw new ArrayIndexOutOfBoundsException();
		}
		return volume.get(index)[lag]; //still throws exception if lag dne.
	}

	/**
	 * Converts a given 1D index into a 3D coordinate.
	 * Returns the 3D coordinate in an int array in the format
	 * {x, y, z}
	 * note: I believe it was agreed upon that this function assumed that the
	 * index was zero based.
	 */
	public int[] convert1Dto3D(int index) {
		int z = index / (mDimensions[0] * mDimensions[1]);
		int remainder = index % (mDimensions[0] * mDimensions[1]);
		int y = remainder / mDimensions[0];
		int x = remainder % mDimensions[0];
		
		return new int[]{x, y, z};
	}
	
	/**
	 * Converts a given 1D index into a view specific coordinates.
	 * Returns the view specific coordinate in an inte array in the format
	 * {x, y, slice}
	 * note: if convert1Dto3D uses 0-based 1D coordinates and index is a 0-based
	 * 1-d coordinate then this returns the proper values.
	 */
	public int[] convert1DtoView(int index, int view) {
		int[] coords = convert1Dto3D(index);
		
		int x = 0, y = 0, slice = 0;
		
		switch (view) {
		case AXIAL:
			x = coords[0];
			y = coords[1];
			slice = coords[2];
			break;
		case SAGITTAL:
			x = coords[1];
			y = coords[2];
			slice = coords[0];
			break;
		case CORONAL:
			x = coords[0];
			y = coords[2];
			slice = coords[1];
			break;
		}
		
		return new int[]{x, y, slice};
	}
	
	/**
	 * Converts a given view coordinate into a 1D index.
	 */
	public int[] convertViewto3D(int x, int y, int slice, int view) {
		
		switch (view) {
		case AXIAL:
			return new int[]{x, y, slice};
		case SAGITTAL:
			return new int[]{slice, x, y};
		case CORONAL:
			return new int[]{x, slice, y};
		};
		
		return null;
	}
	
	/**
	 * Converts a given view coordinate into a 1D index.
	 */
	public int convertViewto1D(int x, int y, int slice, int view) {
		
		switch (view) {
		case AXIAL:
			return convert3Dto1D(x, y, slice);
		case SAGITTAL:
			return convert3Dto1D(slice, x, y);
		case CORONAL:
			return convert3Dto1D(x, slice, y);
		};
		
		return -1;
	}
	
	/**
	 * Converts a given 3D coordinate into a 1D index.
	 */
	public int convert3Dto1D(int x, int y, int z) {
		return (mDimensions[0] * mDimensions[1] * z) + (mDimensions[0] * y) + x;
	}
	
	/**
	 * Returns the slices for the given view.
	 */
	public ArrayList<ArrayList<Integer>> getSlices(int brainView) {
		if (brainView == BrainData.AXIAL)
			return mAxialSlices;
		else if (brainView == BrainData.SAGITTAL)
			return mSagittalSlices;
		else if (brainView == BrainData.CORONAL)
			return mCoronalSlices;
		else
			return null;
	}
	
	/**
	 * Given a view (either axial, sagittal or coronal), returns the width
	 * of a brain slice in this view.
	 */
	public int getWidth(int brainView) {
		switch (brainView) {
		case AXIAL:
			return mDimensions[0];
		case SAGITTAL:
			return mDimensions[1];
		case CORONAL:
			return mDimensions[0];
		default:
			return 0;
		}
	}

	/**
	 * Given a view (either axial, sagittal or coronal), returns the height
	 * of a brain slice in this view.
	 */
	public int getHeight(int brainView) {
		switch (brainView) {
		case AXIAL:
			return mDimensions[1];
		case SAGITTAL:
			return mDimensions[3];
		case CORONAL:
			return mDimensions[3];
		default:
			return 0;
		}
	}
	
	/**
	 * Given a view (either axial, sagittal or coronal), returns the number
	 * of slices in this view.
	 */
	public int getNumSlices(int brainView) {
		switch (brainView) {
		case AXIAL:
			return mDimensions[3];
		case SAGITTAL:
			return mDimensions[0];
		case CORONAL:
			return mDimensions[1];
		default:
			return 0;
		}
	}
	
	/**
	 * Returns the hashmap of data, for those who might need it.
	 */
	public HashMap<Integer, double[]> getData() {
		return mData.get(mCurrLv);
	}
	
	/**
	 * Returns the data for all LVs together.
	 * [lv][coord][lag] -> value
	 */
	public ArrayList<HashMap<Integer, double[]>> getAllData() {
		return mData;
	}

	/**
	 * Calculate the individual thresholds for each lv.
	 * @param data the entire braindata set (all lvs)
	 */
	public void calculateThresholds(double[][] data) {
		//Individual threshold value for each lv is the 95th percentile value
		//when we are looking at either bootstrap or avg zscored data.
		if(identity.equals(BrainData.BOOTSTRAP_STRING) ||
		   identity.equals(BrainData.AVG_ZSCORED_STRING)){
			calculate95th(data);
		}
		//Otherwise calculate the threshold by finding the mean(max,min)*1/3.
		else if(identity.equals(BrainData.BRAINLV_STRING) ||
				identity.equals(BrainData.AVG_CANONICAL_STRING) ||
				identity.equals(BrainData.FULL_DATA_STRING)){
			calculateThirdOfMean();
		}
	}

	public void setPublisher(Publisher publisher) {
		mPublisher = publisher;
		mColourModel.setPublisher(publisher);
	}


	public ColourScaleModel getColourScaleModel() {
		return mColourModel;
	}
	
	public void invertLv(int lv) {
		HashMap<Integer, double[]> lvData = mData.get(lv);
		
		for (double[] values : lvData.values() ) {
			for (int i = 0; i < values.length; ++i) {
				values[i] = -values[i];
			}
		}
		
		double newMax = -mMinValues[lv];
		double newMin = -mMaxValues[lv];
		mMaxValues[lv] = newMax;
		mMinValues[lv] = newMin;
		
		double[] colours = mColourModel.getColourScale(lv);
		newMax = -colours[1];
		newMin = -colours[0];

		mColourModel.toggleInverted(lv);
		mColourModel.setColourScale(lv, newMax, newMin, colours[2]);
		
		mPublisher.publishEvent(new InvertedLvEvent() );
	}


	public void flipHorizontal(int brainView) {
		int width = getWidth(brainView);
		
		for (int i = 0; i < mData.size(); ++i) {
			HashMap<Integer, double[]> newData = new HashMap<Integer, double[]>();
			
			for (Integer coord1D : mData.get(i).keySet() ) {
				double[] data = mData.get(i).get(coord1D);
				
				int[] coordView = convert1DtoView(coord1D, brainView);
				int newX = width - coordView[0];
				
				coord1D = convertViewto1D(newX, coordView[1], coordView[2], brainView);
				newData.put(coord1D, data);
			}
			
			mData.get(i).clear();
			mData.get(i).putAll(newData);
		}
		
		makeBrainSlices();
	}


	public void flipVertical(int brainView) {
		int height = getHeight(brainView);
		
		for (int i = 0; i < mData.size(); ++i) {
			HashMap<Integer, double[]> newData = new HashMap<Integer, double[]>();
			
			for (Integer coord1D : mData.get(i).keySet() ) {
				double[] data = mData.get(i).get(coord1D);
				
				int[] coordView = convert1DtoView(coord1D, brainView);
				int newY = height - coordView[1];
				
				int newCoord = convertViewto1D(coordView[0], newY, coordView[2], brainView);
				newData.put(newCoord, data);
			}
			
			mData.get(i).clear();
			mData.get(i).putAll(newData);
		}
		
		makeBrainSlices();
	}
	
	public static String viewToString(int brainView) {
		switch (brainView) {
		case AXIAL:
			return "Axial";
		case CORONAL:
			return "Coronal";
		case SAGITTAL:
			return "Sagittal";
		}
		
		return null;
	}

	public void setBrainFilter(BrainFilter filter) {
		mBrainFilter = filter;
	}

	/**
	 * Calculate the threshold value for each lv by finding the 95th percentile
	 * value for that lv.
	 * @param data the entire braindata set (all lvs)
	 */
	private void calculate95th(double[][] data){
		for (int i = 0; i < mNumLvs; ++i) {
			Percentile per = new Percentile();
			mThresholds[i] = per.evaluate(MLFuncs.getColumn(data, i), 95);
		}
	}

	/**
	 * Calculate the threshold value for each lv by finding 
	 * 1/3*(mean(abs(max),abs(min)))
	 * where max is the max value observed for that lv, min is the min.
	 */
	private void calculateThirdOfMean(){
		for (int i = 0; i < mNumLvs; ++i) {
			mThresholds[i] = (Math.abs(mMaxValues[i]) +
					          Math.abs(mMinValues[i])) / 6.0;
			if (Math.abs(mMaxValues[i]) < mThresholds[i] ||
				Math.abs(mMinValues[i]) < mThresholds[i]) {
				mThresholds[i] = 0.0;
			}
		}
	}
}