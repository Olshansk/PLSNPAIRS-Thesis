package pls.chrome.result.model;

import java.util.ArrayList;

public class PlsResultModel extends ResultModel {
	private double[][] mSProbability = null;
	private Integer mNumPermutations = null;
	private double[][] mOrigUsc = null;
	private double[][] mUlUsc = null;
	private double[][] mLlUsc = null;
	private ArrayList<ArrayList<String>> mBehavNames = null;
	private double[][] mBehavData = null;
	private ArrayList<ArrayList<String>> mDatamatProfiles = null;
	private ArrayList<ArrayList<String>> mDatamatCorrs = null;

	public void setSProbability(double[][] probability) {
		mSProbability = probability;
	}
	
	public double[][] getSProbability() {
		return mSProbability;
	}

	public Integer getNumPermutations() {
		return mNumPermutations;
	}

	public void setNumPermutations(int numPermutations) {
		mNumPermutations = numPermutations;
	}
	
	
	public void setOrigUsc(double[][] origUsc) {
		mOrigUsc = origUsc;
	}
	
	public double[][] getOrigUsc() {
		return mOrigUsc;
	}
	
	public void setUlUsc(double[][] ulUsc) {
		mUlUsc = ulUsc;
	}
	
	public double[][] getUlUsc() {
		return mUlUsc;
	}
	
	public void setLlUsc(double[][] llUsc) {
		mLlUsc = llUsc;
	}
	
	public double[][] getllUsc() {
		return mLlUsc;
	}
	
	public void setBehavNames(ArrayList<ArrayList<String>> behavNames) {
		mBehavNames = behavNames;
	}
	
	public ArrayList<ArrayList<String>> getBehavNames() {
		return mBehavNames;
	}

	public void setBehavData(double[][] behavData) {
		mBehavData = behavData;
	}
	
	public double[][] getBehavData() {
		return mBehavData;
	}	
	
	public void setDatamatProfiles(ArrayList<ArrayList<String>> datamatProfiles) {
		mDatamatProfiles = datamatProfiles;
	}
	
	public ArrayList<ArrayList<String>> getDatamatProfiles() {
		return mDatamatProfiles;
	}
	
	public void setDatamatCorrs(ArrayList<ArrayList<String>> datamatCorrsList) {
		mDatamatCorrs = datamatCorrsList;
	}
	
	public ArrayList<ArrayList<String>> getDatamatCorrs() {
		return mDatamatCorrs;
	}

	protected void addToRepository(GeneralRepository repository) {
		repository.addModel(getFilename(), this);
	}

	protected void removeFromRepository(GeneralRepository repository) {
		repository.removePlsModel(getFilename() );
	}
	
	public String getVariableType() {
		return "Latent Variable";
	}
	
	public String getAbbrVariableType() {
		return "LV";
	}
}
