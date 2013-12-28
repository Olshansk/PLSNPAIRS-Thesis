package pls.chrome.result.model;

import java.util.ArrayList;

public class NPairsResultModel extends ResultModel {
	
	private double[][] mCvScores = null;
	private double[] mClassLabels = null;
	private ArrayList<String> mClassNames = null;
	private double[][] mEvals1 = null;
	private double[][] mCvScoresTrain = null;
	private double[][] mEvals2 = null;
	private double[][] mCvScoresTest = null;
	private double[][] mReprodCC = null;
	private double[][] mPPTrueClass = null;
	private double[][] mSplitVols = null;
	private double[] mSplitObjLabels = null;
	private String mSplitType; //split object type name i.e 'Run/Session/etc'

	public void setCvScores(double[][] array) {
		mCvScores = array;
	}
	
	public double[][] getCvScores() {
		return mCvScores;
	}

	public void setClassLabels(double[] column) {
		mClassLabels = column;
	}
	
	public double[] getClassLabels() {
		return mClassLabels;
	}

	public void setClassNames(ArrayList<String> classNames) {
		mClassNames = classNames;
	}
	
	public ArrayList<String> getClassNames() {
		return mClassNames;
	}

	public void setEvals1(double[][] array) {
		mEvals1 = array;
	}
	
	public double[][] getEvals1() {
		return mEvals1;
	}

	public void setCvScoresTrain(double[][] array) {
		mCvScoresTrain = array;
	}
	
	public double[][] getCvScoresTrain() {
		return mCvScoresTrain;
	}

	public void setEvals2(double[][] array) {
		mEvals2 = array;
	}
	
	public double[][] getEvals2() {
		return mEvals2;
	}

	public void setCvScoresTest(double[][] array) {
		mCvScoresTest = array;
	}
	
	public double[][] getCvScoresTest() {
		return mCvScoresTest;
	}

	public void setReprodCC(double[][] array) {
		mReprodCC = array;
	}
	
	public double[][] getReprodCC() {
		return mReprodCC;
	}

	public void setPPTrueClass(double[][] array) {
		mPPTrueClass = array;
	}
	
	public double[][] getPPTrueClass() {
		return mPPTrueClass;
	}
	
	protected void addToRepository(GeneralRepository repository) {
		repository.addModel(getFilename(), this);
	}

	protected void removeFromRepository(GeneralRepository repository) {
		repository.removeNPairsModel(getFilename());
	}
	
	public String getVariableType() {
		return "Canonical Variable";
	}
	
	public String getAbbrVariableType() {
		return "CV";
	}

	public void setSplitVols(double[][] mSplitVols) {
		this.mSplitVols = mSplitVols;
	}

	public double[][] getSplitVols() {
		return mSplitVols;
	}

	public void setSplitObjLabels(double[] mSplitObjLabels) {
		this.mSplitObjLabels = mSplitObjLabels;
	}

	public double[] getSplitObjLabels() {
		return mSplitObjLabels;
	}
	
	public void setSplitType(String splitType){
		mSplitType = splitType;
	}
	
	public String getSplitType(){
		return mSplitType;
	}

}
