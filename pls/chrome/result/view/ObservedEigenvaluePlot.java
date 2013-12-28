package pls.chrome.result.view;

import java.awt.event.ActionEvent;

import javax.swing.JComboBox;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.labels.StandardCategoryItemLabelGenerator;
import org.jfree.chart.labels.StandardXYItemLabelGenerator;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.CategoryDataset;
import org.jfree.data.general.DatasetUtilities;

import pls.shared.GlobalVariablesFunctions;
import pls.chrome.result.model.ChartFactoryChild;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.NPairsResultModel;
import pls.chrome.result.model.ResultModel;
import pls.shared.MLFuncs;

@SuppressWarnings("serial")
public class ObservedEigenvaluePlot extends AbstractPlot {
	
	public static final int FULL_DATA = 0;
	public static final int SPLIT_1 = 1;
	public static final int SPLIT_2 = 2;
	
	private JComboBox mPlotList;

	private JFreeChart fullDataChart;
	private JFreeChart split1Chart;
	private JFreeChart split2Chart;
	
	public ObservedEigenvaluePlot(String title, GeneralRepository repository) {
		super(title, repository, GlobalVariablesFunctions.NPAIRS);
		
		String[] plotTypes = new String[3];
		plotTypes[FULL_DATA] = "Full-data Reference";
		plotTypes[SPLIT_1] = "Split 1";
		plotTypes[SPLIT_2] = "Split 2";
		
		mPlotList = new JComboBox(plotTypes);
		mFilePanel.add(mPlotList);

		mPlotList.addActionListener(this);
	}
	
	public double[][] getRangeData(ResultModel model) {
		NPairsResultModel newModel = (NPairsResultModel) model;
		
		// Retrieves the min and max range global range values of the
		// three data types here instead of combining them into one
		// large single 2-d array to do the same thing. The data
		// does not need to be stored for this class anyway.
		getMinMaxValues(newModel.getS(), GLOBAL, RANGE);
		getMinMaxValues(newModel.getEvals1(), GLOBAL, RANGE);
		getMinMaxValues(newModel.getEvals2(), GLOBAL, RANGE);
		
		return null;
	}
	
	public void makeChart(int fileIndex) {
		localRangeMin = Double.POSITIVE_INFINITY;
		localRangeMax = Double.NEGATIVE_INFINITY;

		NPairsResultModel model = mRepository.getNpairsModel(mResultFilePaths.get(fileIndex) );;
		String abbrVariableType = model.getAbbrVariableType();
		String variableType = model.getVariableType() + "s";
		
		// Retrieves the overall max and min local range values of the three
		// data types first.
		getMinMaxValues(model.getS(), LOCAL, RANGE);
		getMinMaxValues(model.getEvals1(), LOCAL, RANGE);
		getMinMaxValues(model.getEvals2(), LOCAL, RANGE);
		
		// Creates the charts for the three data types.
		CategoryDataset fullDataset = createDataset(model.getS(), abbrVariableType);
		CategoryDataset split1Dataset = createDataset(model.getEvals1(), abbrVariableType);
		CategoryDataset split2Dataset = createDataset(model.getEvals2(), abbrVariableType);
		
		fullDataChart = createChart(fullDataset, variableType);
		split1Chart = createChart(split1Dataset, variableType);
		split2Chart = createChart(split2Dataset, variableType);
		
		// Applies the overall max and min range values to all three charts.
		if (mGlobalButton.isSelected()) {
			applyMinMaxValues(fullDataChart, GLOBAL, RANGE);
			applyMinMaxValues(split1Chart, GLOBAL, RANGE);
			applyMinMaxValues(split2Chart, GLOBAL, RANGE);
		} else if (mLocalButton.isSelected()) {
			applyMinMaxValues(fullDataChart, LOCAL, RANGE);
			applyMinMaxValues(split1Chart, LOCAL, RANGE);
			applyMinMaxValues(split2Chart, LOCAL, RANGE);
		}
		
		if (mPlotList != null) {
			mPlotList.setSelectedIndex(FULL_DATA);
		}
		mChartPanel.setChart(fullDataChart);
	}
	
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == mPlotList) {
			int selected = mPlotList.getSelectedIndex();
			if (selected == FULL_DATA) {
				mChartPanel.setChart(fullDataChart);
			} else if (selected == SPLIT_1) {
				mChartPanel.setChart(split1Chart);
			} else {
				mChartPanel.setChart(split2Chart);
			}
		} else {
			super.actionPerformed(e);
		}
	}
	
	private CategoryDataset createDataset(double[][] data, String variableType) {
		return DatasetUtilities.createCategoryDataset(
				"Value", variableType, MLFuncs.transpose(data));
	}
	
	private JFreeChart createChart(CategoryDataset dataset, String variableType) {
		JFreeChart chart = ChartFactoryChild.createStackedBarChart(
				"Observed Eigenvalue Plot", variableType,
				"Observed Eigenvalues", dataset,
				PlotOrientation.VERTICAL, false, true, false);
		
		chart.getCategoryPlot().getRenderer().setBaseItemLabelsVisible(true);
		chart.getCategoryPlot().getRenderer().setBaseItemLabelGenerator(new StandardCategoryItemLabelGenerator() );
		
		return chart;
	}
	
	protected boolean ModelIsApplicable(ResultModel model) {
		return model.getS() != null;
	}
}