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

@SuppressWarnings("serial")
public class ObservedSingularValuesPlot extends AbstractPlot {
	
	public ObservedSingularValuesPlot(String title, GeneralRepository repository) {
		super(title, repository, GlobalVariablesFunctions.PLS);
		
		mButtonPanel.remove(mLocalButton);
	}
	
	public void makeChart(int fileIndex) {
		ResultModel model = mRepository.getGeneral(mResultFilePaths.get(fileIndex) );;
		double[][] data = mRangeData.get(fileIndex);
		
		String abbrVariableType = model.getAbbrVariableType().toUpperCase();
		String variableType = model.getVariableType() + "s";
		
		CategoryDataset svDataset = DatasetUtilities.createCategoryDataset(
				"Value", abbrVariableType, data);
		JFreeChart svChart = ChartFactory.createStackedBarChart(
				"Observed Singular Value Plot", variableType,
				"Observed Singular Values", svDataset,
				PlotOrientation.VERTICAL, false, true, false);
		
		svChart.getCategoryPlot().getRenderer().setBaseItemLabelsVisible(true);
		svChart.getCategoryPlot().getRenderer().setBaseItemLabelGenerator(new StandardCategoryItemLabelGenerator() );
		
		if (mGlobalButton.isSelected()) {
			applyMinMaxValues(svChart, GLOBAL, RANGE);
		}
		mChartPanel.setChart(svChart);
	}
	
	public double[][] getRangeData(ResultModel model) {
		return MLFuncs.transpose(model.getS());
	}
	
	protected boolean ModelIsApplicable(ResultModel model) {
		return model.getS() != null;
	}
}
