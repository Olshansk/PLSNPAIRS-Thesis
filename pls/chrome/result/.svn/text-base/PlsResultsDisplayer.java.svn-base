package pls.chrome.result;

import java.util.ArrayList;

import javax.swing.JTabbedPane;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.CategoryDataset;
import org.jfree.data.general.DatasetUtilities;

import pls.chrome.MainFrame;
import pls.chrome.result.model.BrainData;
import pls.chrome.result.model.PlsResultModel;
import pls.chrome.result.model.ResultModel;
import pls.chrome.result.view.AbstractPlot;
import pls.chrome.result.view.BrainDesignScoresPlot;
import pls.chrome.result.view.DesignLatentVariablesPlot;
import pls.chrome.result.view.DesignScoresPlot;
import pls.chrome.result.view.ResponseFunctionPlot;
import pls.chrome.result.view.ResultContrastPanel;
import pls.chrome.result.view.TemporalBrainCorrelationPlot;
import pls.chrome.result.view.TemporalBrainScoresPlot;
import pls.chrome.result.view.VoxelIntensityResponsePlot;
import pls.shared.MLFuncs;
import Jama.Matrix;

/**
 * A class used to parse data stored in a results mat-lab file and create
 * various plots and charts to display the results. Also manages a progress
 * dialog enabling the user to track the progress of content creation.
 * Update: This class is not in use as far as I can tell -Fletcher
 */
class PlsResultsDisplayer extends AbstractResultsDisplayer {
	
	private PlsResultModel mPlsResultModel = null;
//	private PlsRepository mPlsRepository = null;

	public PlsResultsDisplayer(DetachableTabbedPane tabs, String fileName) {
		super(tabs, fileName);
	}

	protected void loadOtherPlots() {
		
		// Create design latent variables plot
//		if (mResultModel.getDesignLv() != null) {
//			String variableType = mRepository.getGeneral().getVariableType();
//			progress.appendMessage("Creating Design " + variableType + "s ...");
//			DesignLatentVariablesPlot dlvp = new DesignLatentVariablesPlot(mTabbedPane, "Design " + variableType + "s", mRepository);
//			mTabbedPane.addTab("Design " + variableType + "s", dlvp);
//			progress.updateStatus(1);
//		}
		
		// Plot permuted singular values
//		if (mPlsResultModel.getSProbability() != null &&
//				mPlsResultModel.getNumPermutations() != null) {
//			progress.appendMessage("Creating Permuted Singular Values ...");
//			
//			@SuppressWarnings("serial")
//			AbstractPlot psvPlot = new AbstractPlot(mTabbedPane, "Permuted Singular Values", mRepository, MainFrame.PLS) {
//				public void makeChart(int fileIndex) {
//					PlsResultModel model  = mRepository.getPlsModel(mResultFilePaths.get(fileIndex) );;
//					double[][] data = mChartData.get(fileIndex);
//					
//					String abbrVariableType = model.getAbbrVariableType().toUpperCase();
//					String variableType = model.getVariableType() + "s";
//					
//					if (data != null) {
//						CategoryDataset psvDataset = DatasetUtilities.createCategoryDataset(
//								"Value", abbrVariableType, data);
//						JFreeChart psvChart = ChartFactory.createStackedBarChart(
//						"Permuted values greater than observed, " + model.getNumPermutations()
//								+ " permutation tests", variableType,
//						"Probability", psvDataset, PlotOrientation.VERTICAL, false,
//						true, false);
//						
//						if (mGlobalButton.isSelected()) {
//							applyMinMaxRangeValues(psvChart, GLOBAL);
//						}
//						mChartPanel.setChart(psvChart);
//					}
//				}
//				
//				public double[][] getChartData(ResultModel model) {
//					return ((PlsResultModel) model).getSProbability();
//				}
//			};
//	
//			mTabbedPane.addTab("Permuted Singular Values", psvPlot);
//			progress.updateStatus(1);
//		}

		
		// Create contrasts plot
//		String contrastFile = mResultModel.getConstrastFilename(); 
//		if (contrastFile != null && !contrastFile.equals("NONE")
//				&& !contrastFile.equals("MULTIBLOCK")
//				&& !contrastFile.equals("BEHAV")) {
//			progress.appendMessage("Creating Contrasts Information ...");
//			mTabbedPane.addTab("Contrasts Information", new ResultContrastPanel(mTabbedPane, "Contrasts Information", mRepository));
//			progress.updateStatus(1);
//		}
//		
//		double[][] s = mResultModel.getS(); 
//		if (s != null) {
//			// Plot observed singular values
//			progress.appendMessage("Creating Observed Singular Value Plot ...");
//			
//			@SuppressWarnings("serial")
//			AbstractPlot svPlot = new AbstractPlot(mTabbedPane, "Observed Singular Value Plot", mRepository, MainFrame.PLS) {
//				public void makeChart(int fileIndex) {
//					ResultModel model = mRepository.getGeneral(mResultFilePaths.get(fileIndex) );;
//					double[][] data = mChartData.get(fileIndex);
//					
//					String abbrVariableType = model.getAbbrVariableType().toUpperCase();
//					String variableType = model.getVariableType() + "s";
//					
//					CategoryDataset svDataset = DatasetUtilities.createCategoryDataset(
//							"Value", abbrVariableType, data);
//					JFreeChart svChart = ChartFactory.createStackedBarChart(
//							"Observed Singular Value Plot", variableType,
//							"Observed Singular Values", svDataset,
//							PlotOrientation.VERTICAL, false, true, false);
//					
//					if (mGlobalButton.isSelected()) {
//						applyMinMaxRangeValues(svChart, GLOBAL);
//					}
//					mChartPanel.setChart(svChart);
//				}
//				
//				public double[][] getChartData(ResultModel model) {
//					return MLFuncs.transpose(model.getS());
//				}
//			};
//			
//			mTabbedPane.addTab("Observed Singular Value Plot", svPlot);
//			progress.updateStatus(1);
//
//			// Plot percent crossblock
//			progress.appendMessage("Creating Percent Crossblock Covariance ...");
//			
//			@SuppressWarnings("serial")
//			AbstractPlot pccPlot = new AbstractPlot(mTabbedPane, "Percent Crossblock Covariance", mRepository, MainFrame.PLS) {
//				public void makeChart(int fileIndex) {
//					ResultModel model = mRepository.getGeneral(mResultFilePaths.get(fileIndex) );;
//					double[][] data = mChartData.get(fileIndex);
//					
//					String abbrVariableType = model.getAbbrVariableType().toUpperCase();
//					String variableType = model.getVariableType() + "s";
//					
//					CategoryDataset pccDataset = DatasetUtilities
//					.createCategoryDataset("Value", abbrVariableType, data);
//					JFreeChart pccChart = ChartFactory.createStackedBarChart(
//					"Percent Crossblock Covariance", variableType,
//					"Percent", pccDataset, PlotOrientation.VERTICAL, false,
//					true, false);
//					
//					if (mGlobalButton.isSelected()) {
//						applyMinMaxRangeValues(pccChart, GLOBAL);
//					}
//					mChartPanel.setChart(pccChart);
//				}
//				
//				public double[][] getChartData(ResultModel model) {
//					double[][] s = model.getS();
//					double[][] percentage;
//					if (s.length == s[0].length || s[0].length > 1) {
//						Matrix Ssq = MLFuncs.square(MLFuncs.diag(new Matrix(s)));
//						percentage = Ssq.times(1.0 / MLFuncs.sum(Ssq) * 100)
//							.transpose().getArray();
//					} else {
//						Matrix Ssq = MLFuncs.square(new Matrix(s));
//						percentage = Ssq.times((1.0 / MLFuncs.sum(Ssq)) * 100)
//							.transpose().getArray();
//					}
//					
//					return percentage;
//				}
//			};
//			
//			mTabbedPane.addTab("Percent Crossblock Covariance", pccPlot);
//			progress.updateStatus(1);
//		}
		
		// Create brain vs design scores plot
//		if (mResultModel.getDesignScores() != null && mResultModel.getBrainScores() != null) {
//			progress.appendMessage("Creating Brain vs Design Scores ...");
//			 mTabbedPane.addTab("Brain vs Design Scores", new BrainDesignScoresPlot(mTabbedPane, "Brain vs Design Scores", mRepository));
//			progress.updateStatus(1);
//		}
		
		// If this is block, make voxel intensity response
//		if (mResultModel.getWindowSize() > 1) {
//			progress.appendMessage("Creating Response Function Plot ...");
//			mTabbedPane.addTab("Response Function Plot", new ResponseFunctionPlot(mTabbedPane, "Response Function Plot", mRepository));
//			progress.updateStatus(1);
//		}
//		else {
//			progress.appendMessage("Creating Voxel Intensity Response Plot ...");
//			mTabbedPane.addTab();
//			progress.updateStatus(1);
//		}

		// Create temporal brain scores plot
//		if (mResultModel.getWindowSize() > 1 && 
//				contrastFile != null && !contrastFile.equals("BEHAV")) {
//			progress.appendMessage("Creating Temporal Brain Scores Plot ...");
//			progress.requestFocus();
//			mTabbedPane.addTab("Temporal Brain Scores Plot", new TemporalBrainScoresPlot(mTabbedPane, "Temporal Brain Scores Plot", mRepository));
//			progress.updateStatus(1);
//		}
		
		// Create temporal brain correlation plot
//		if (mResultModel.getWindowSize() > 1 && 
//				contrastFile != null && contrastFile.equals("BEHAV")
//				&& mPlsResultModel.getBehavNames() != null &&
//				mPlsResultModel.getBehavData() != null) {
//			progress.appendMessage("Creating Temporal Brain Correlation Plot ...");
//			progress.requestFocus();
//			mTabbedPane.addTab("Temporal Brain Correlation Plot", new TemporalBrainCorrelationPlot(mTabbedPane, "Temporal Brain Correlation Plot", mRepository));
//			progress.updateStatus(1);
//		}
		
		// Create design scores plot
//		if (mResultModel.getDesignScores() != null && mPlsResultModel.getSProbability() != null) {
//			progress.appendMessage("Creating Design Scores ...");
//			mTabbedPane.addTab("Design Scores", new DesignScoresPlot(mTabbedPane, "Design Scores", mRepository));
//			progress.updateStatus(1);
//		}
	}
	
	protected ArrayList<String> getDataTypes() {
		ArrayList<String> dataTypes = new ArrayList<String>();
		dataTypes.add(BrainData.BOOTSTRAP_STRING);
		dataTypes.add(BrainData.BRAINLV_STRING);
		
		return dataTypes;
	}

	protected void createResultModel() {
		// Load the model
		PlsResultLoader loader = new PlsResultLoader(mFileName);
		
		try {
			loader.loadFile();
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		
		mPlsResultModel = loader.getPlsResultModel();
		mResultModel = loader.getResultModel();
		
		// Put the model into the repository
		mRepository.addModel(mFileName, mPlsResultModel);
		mRepository.getControlPanelModel().initModel();
		mRepository.getPlotManager().refreshPlots();
	}
}