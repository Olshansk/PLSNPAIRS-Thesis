package pls.chrome.result.view;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.labels.StandardCategoryItemLabelGenerator;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.CategoryDataset;
import org.jfree.data.general.DatasetUtilities;

import pls.shared.GlobalVariablesFunctions;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.ResultModel;
import pls.shared.MLFuncs;
import Jama.Matrix;

@SuppressWarnings("serial")
public class PercentCrossblockCovariancePlot extends AbstractPlot {
	public PercentCrossblockCovariancePlot(String title, GeneralRepository repository) {
		super(title, repository, GlobalVariablesFunctions.PLS);
		
		mButtonPanel.remove(mLocalButton);
	}
	
	public void makeChart(int fileIndex) {
		ResultModel model = mRepository.getGeneral(mResultFilePaths.get(fileIndex) );;
		double[][] data = mRangeData.get(fileIndex);
		
		String abbrVariableType = model.getAbbrVariableType().toUpperCase();
		String variableType = model.getVariableType() + "s";
		
		CategoryDataset pccDataset = DatasetUtilities
		.createCategoryDataset("Value", abbrVariableType, data);
		JFreeChart pccChart = ChartFactory.createStackedBarChart(
		"Percent Crossblock Covariance", variableType,
		"Percent", pccDataset, PlotOrientation.VERTICAL, false,
		true, false);
		
		pccChart.getCategoryPlot().getRenderer().setBaseItemLabelsVisible(true);
		pccChart.getCategoryPlot().getRenderer().setBaseItemLabelGenerator(new StandardCategoryItemLabelGenerator() );
		
		if (mGlobalButton.isSelected()) {
			applyMinMaxValues(pccChart, GLOBAL, RANGE);
		}
		mChartPanel.setChart(pccChart);
	}
	
	public double[][] getRangeData(ResultModel model) {
		double[][] s = model.getS();
		double[][] percentage;
		if (s.length == s[0].length || s[0].length > 1) {
			Matrix Ssq = MLFuncs.square(MLFuncs.diag(new Matrix(s)));
			percentage = Ssq.times(1.0 / MLFuncs.sum(Ssq) * 100)
				.transpose().getArray();
		} else {
			Matrix Ssq = MLFuncs.square(new Matrix(s));
			percentage = Ssq.times((1.0 / MLFuncs.sum(Ssq)) * 100)
				.transpose().getArray();
		}
		
		return percentage;
	}
	
	protected boolean ModelIsApplicable(ResultModel model) {
		return model.getS() != null;
	}
}
