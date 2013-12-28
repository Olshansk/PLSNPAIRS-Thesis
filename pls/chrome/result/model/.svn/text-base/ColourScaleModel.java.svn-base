package pls.chrome.result.model;

import java.util.ArrayList;

import pls.chrome.result.controller.Publisher;
import pls.chrome.result.controller.observer.colourscale.ColourScaleEvent;

public class ColourScaleModel {
	private ArrayList<Boolean> mInvertValues;
	private ArrayList<Double> mMaxValues;
	private ArrayList<Double> mMinValues;
	private ArrayList<Double> mThreshValues;
	
	private Publisher mPublisher;
	private BrainData mParent;
	
	private int mCurrLvNum = 0;
	
	public ColourScaleModel(BrainData parent) {
		mParent = parent;
		
		int numLvs = mParent.getNumLvs();
		
		mInvertValues = new ArrayList<Boolean>(numLvs);
		mMaxValues = new ArrayList<Double>(numLvs);
		mMinValues = new ArrayList<Double>(numLvs);
		mThreshValues = new ArrayList<Double>(numLvs);
		
		for (int i = 0; i < numLvs; ++i) {
			mInvertValues.add(false);
			double[] colours = mParent.getMaxMinThresh(i);
			mMaxValues.add(colours[0]);
			mMinValues.add(colours[1]);
			mThreshValues.add(colours[2]);
		}
	}
	
	protected void setPublisher(Publisher pub) {
		mPublisher = pub;
	}
	
	public void setColourScale(int lv, double max, double min, double threshold) {
		mMaxValues.set(lv, max);
		mMinValues.set(lv, min);
		mThreshValues.set(lv, threshold);
		
		mPublisher.publishEvent(new ColourScaleEvent() );
	}
	
	public void setColourScale(double max, double min, double threshold) {
		mMaxValues.set(mCurrLvNum, max);
		mMinValues.set(mCurrLvNum, min);
		mThreshValues.set(mCurrLvNum, threshold);
		
		mPublisher.publishEvent(new ColourScaleEvent() );
	}
	
	public double[] getColourScale(int lvNum) {	
		double max = mMaxValues.get(lvNum);
		double min = mMinValues.get(lvNum);
		double thresh = mThreshValues.get(lvNum);
		
		return new double[]{max, min, thresh};
	}
	
	/**
	 * Returns the max, min and threshold of the currently selected data type
	 * and lv in the format {max, min, threshold}
	 */
	public double[] getColourScale() {
		double max = mMaxValues.get(mCurrLvNum);
		double min = mMinValues.get(mCurrLvNum);
		double thresh = mThreshValues.get(mCurrLvNum);
		
		return new double[]{max, min, thresh};
	}

	public boolean isInverted() {
		return mInvertValues.get(mCurrLvNum);
	}
	
	public boolean isInverted(int lvNum) {
		return mInvertValues.get(lvNum);
	}
	
	public void toggleInverted(int lvNum) {
		mInvertValues.set(lvNum, !mInvertValues.get(lvNum) );
	}
	
	public void toggleInverted() {
		toggleInverted(mCurrLvNum);
	}
	
	protected void setCurrLv(int lvNum) {
		mCurrLvNum = lvNum;
	}
}
