package pls.chrome.result.view;

import java.util.HashMap;
import java.util.Vector;

import Jama.Matrix;

import pls.shared.MLFuncs;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.PlsResultModel;
import pls.chrome.result.model.ResultModel;

@SuppressWarnings("serial")
public class TemporalBrainCorrelationPlot extends TemporalBrainScoresPlot {
	
	private double[][] mBehavData;

	public TemporalBrainCorrelationPlot(String title, GeneralRepository repository) {
		super(title, repository, true);
		
		mPlotType = "Brain Correlation";
	}
	
	protected void updateLvComboBoxAndTabs(int fileIndex) {
		super.updateLvComboBoxAndTabs(fileIndex);
		PlsResultModel model = mRepository.getPlsModel(mResultFilePaths.get(fileIndex));

		mBehavData = model.getBehavData();
	}
	
	protected Vector<Vector<double[]>> calculateData(int group, int lv) {
		int winSize = st.winSize;
		int numConditions = st.conditions.length;
		int numVoxels = st.numVoxels;
		int numEvents = st.eventList.length;
		int numBehav = mBehavData[0].length;
		
		// Shove all the data into a 3-D array	
		double[][][] blv = new double[winSize][numVoxels][1];
		HashMap<Integer, double[]> currLv = mBrainLv.get(lv);
		for (int i = 0; i != mCoords.size(); i++) {
			double[] lags = currLv.get(mCoords.get(i) - 1);
			
			for (int j = 0; j != lags.length; j++) {
				blv[j][i][0] = lags[j];
			}
		}
		
		// Calculating the output
		double[][][] brainScores = new double[winSize][numEvents][1];
		double[][][] bcorr = new double[winSize][numConditions * numBehav][1];
		for (int i = 0; i != winSize; i++) {
			Matrix datamatMatrix = new Matrix(MLFuncs.getXYColumns(mDatamat, i));
			Matrix blvMatrix = new Matrix(blv[i]);
			brainScores[i] = datamatMatrix.times(blvMatrix).getArray();
			
			bcorr[i] = MLFuncs.rri_corr_maps(mBehavData, brainScores[i], st.numSubjectList[0], numConditions);
		}
		
		Vector<Vector<double[]>> result = new Vector<Vector<double[]>>();
		
		int currSubject = 0;
		for (int i = 0; i != group; i++) {
			currSubject += numConditions * numBehav;
		}
		
		for (int j = 0; j < numConditions; j++) {
			Vector<double[]> currConditionResult = new Vector<double[]>();
			double[][] thisConditionValues = new double[numBehav][winSize];
			
			for (int k = 0; k < numBehav; k++) {
				double[] plotData = new double[winSize];
				for (int m = 0; m != winSize; m++) {
					plotData[m] = bcorr[m][currSubject][0];
				}
						
				currConditionResult.add(plotData);
				thisConditionValues[k] = plotData;
					
				currSubject++;
			}
					
			double[] columnMean = MLFuncs.columnMean(thisConditionValues)[0];
			currConditionResult.add(columnMean);
			result.add(currConditionResult);
		}

		return result;
	}

	protected int getNumSubjects(int group) {
		
		// Here, the number of behav values is used instead and it
		// is the same no matter what group is given.
		return mBehavData[0].length;
	}
	
	protected boolean ModelIsApplicable(ResultModel model) {
		//String contrastFile = ((PlsResultModel)model).getConstrastFilename();
		
		//return model.getWindowSize() > 1 && contrastFile != null && contrastFile.equals("BEHAV");
		return model.getWindowSize() > 1 && ((PlsResultModel)model).getBehavData() != null;
	}
	
}