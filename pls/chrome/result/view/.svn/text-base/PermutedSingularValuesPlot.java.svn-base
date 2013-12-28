package pls.chrome.result.view;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.CategoryDataset;
import org.jfree.data.general.DatasetUtilities;

import pls.shared.GlobalVariablesFunctions;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.PlsResultModel;
import pls.chrome.result.model.ResultModel;

@SuppressWarnings("serial")
public class PermutedSingularValuesPlot extends AbstractPlot {
	public PermutedSingularValuesPlot(String title, GeneralRepository repository) {
		super(title, repository, GlobalVariablesFunctions.PLS);
		
		mButtonPanel.remove(mLocalButton);
	}
		
	public void makeChart(int fileIndex) {
		PlsResultModel model  = mRepository.getPlsModel(mResultFilePaths.get(fileIndex) );;
		double[][] data = mRangeData.get(fileIndex);
		
		String abbrVariableType = model.getAbbrVariableType().toUpperCase();
		String variableType = model.getVariableType() + "s";
		
		if (data != null) {
			CategoryDataset psvDataset = DatasetUtilities.createCategoryDataset(
					"Value", abbrVariableType, data);
			JFreeChart psvChart = ChartFactory.createStackedBarChart(
			"Permuted values greater than observed, " + model.getNumPermutations()
					+ " permutation tests", variableType,
			"Probability", psvDataset, PlotOrientation.VERTICAL, false,
			true, false);
			
			if (mGlobalButton.isSelected()) {
				applyMinMaxValues(psvChart, GLOBAL, RANGE);
			}
			mChartPanel.setChart(psvChart);
		}
	}
	
	public double[][] getRangeData(ResultModel model) {
		return ((PlsResultModel) model).getSProbability();
	}
	
	protected boolean ModelIsApplicable(ResultModel model) {
		PlsResultModel plsModel = (PlsResultModel)model;
		return plsModel.getSProbability() != null &&
		plsModel.getNumPermutations() != null;
	}
}
