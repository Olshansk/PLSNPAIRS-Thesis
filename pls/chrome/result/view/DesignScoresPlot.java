package pls.chrome.result.view;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.event.ActionEvent;

import java.text.DecimalFormat;
import java.util.ArrayList;

import javax.swing.JComboBox;
import javax.swing.JTabbedPane;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.DefaultKeyedValues;
import org.jfree.data.category.CategoryDataset;
import org.jfree.data.general.DatasetUtilities;

import Jama.Matrix;

import pls.shared.GlobalVariablesFunctions;
import pls.chrome.result.controller.observer.Event;
import pls.chrome.result.controller.observer.datachange.*;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.PlsResultModel;
import pls.chrome.result.model.ResultModel;
import pls.shared.MLFuncs;

@SuppressWarnings("serial")
public class DesignScoresPlot extends AbstractPlot implements DataChangeObserver{
	
	private double[][] mPercentage;
	private double[][] mProbability;
	
	private JComboBox mLvComboBox = new JComboBox();
	private ChartPanel[] mDesignScoresChartPanel;
	private JTabbedPane mTabs = new JTabbedPane();
	
	public DesignScoresPlot(String title, GeneralRepository repository) {
		super(title, repository, GlobalVariablesFunctions.PLS);
		
		remove(mChartPanel);
		mFilePanel.add(mLvComboBox);
		add(mTabs, BorderLayout.CENTER);
		mRepository.getPublisher().registerObserver(this);
	}
	
	public double[][] getRangeData(ResultModel model) {
		return model.getDesignScores();
	}
	
	protected void updateLvComboBoxAndTabs(int fileIndex) {
		ResultModel model = mRepository.getGeneral(mResultFilePaths.get(fileIndex) );;
		PlsResultModel newModel = (PlsResultModel) model;
		
		int numLvs = newModel.getBrainData().getNumLvs();
		String variableType = newModel.getVariableType();
		
		// Adds a dropdown menu for selecting a different LV.
		if (mLvComboBox == null) {
			mLvComboBox = new JComboBox();
		}
		mLvComboBox.removeActionListener(this);
		mLvComboBox.removeAllItems();
		
		for (int i = 0; i < numLvs; i++) {
			mLvComboBox.addItem(variableType + " #" + (i + 1));
		}
		mLvComboBox.addActionListener(this);
		
		// Creates tabs where each tab represents one group.
		int numGroups = newModel.getNumSubjectList().length;

		mDesignScoresChartPanel = new ChartPanel[numGroups];
		if (mTabs == null) {
			mTabs = new JTabbedPane();
		}
		mTabs.removeAll();
		for (int i = 0; i < numGroups; i++) {
			mDesignScoresChartPanel[i] = new ChartPanel(null);
			Dimension d = mDesignScoresChartPanel[i].getPreferredSize();
			d.height = d.height / 2;
			mDesignScoresChartPanel[i].setPreferredSize(d);
			mTabs.addTab("Group " + (i + 1), mDesignScoresChartPanel[i]);
		}
		
		// Retrieves the probability and percentage data from the given
		// model used to generate the charts.
		mProbability = newModel.getSProbability();
		
		double[][] s = newModel.getS();
		//if s is square or has more than one column
		if (s.length == s[0].length || s[0].length > 1) {
			Matrix Ssq = MLFuncs.square(MLFuncs.diag(new Matrix(s)));
			mPercentage = Ssq.times(1.0 / MLFuncs.sum(Ssq) * 100)
					.transpose().getArray();
		} else {
			//this case happens on both grouped results files I ran
			Matrix Ssq = MLFuncs.square(new Matrix(s));
			mPercentage = Ssq.times((1.0 / MLFuncs.sum(Ssq)) * 100)
					.transpose().getArray();
		}
	}
	
	public void makeChart(int fileIndex) {
		localRangeMin = Double.POSITIVE_INFINITY;
		localRangeMax = Double.NEGATIVE_INFINITY;
		
		// Retrieves the overall min and max values for the data
		// of the model of the given file index.
		double[][] designScores = mRangeData.get(fileIndex);
		getMinMaxValues(designScores, LOCAL, RANGE);
		
		// Uses the first LV as the default selected LV.
		makeLvChart(fileIndex, 0);
	}
	
	public void makeLvChart(int fileIndex, int c) {
		ResultModel model = mRepository.getGeneral(mResultFilePaths.get(fileIndex));
		double[][] designScores = mRangeData.get(fileIndex);
		
		ArrayList<String> conditionNames = model.getConditionNames();
		int[] groups = model.getNumSubjectList();
		String abbrVariableType = model.getAbbrVariableType().toUpperCase();
		
		int numConditions = conditionNames.size();

		double[] designScoresData = MLFuncs.getColumn(designScores, c);
		for (int i = 0; i < groups.length; i++) {
			
			int numSubjects = groups[i];
			DefaultKeyedValues designScoresKeyedData = new DefaultKeyedValues();
			
			for (int j = i * numConditions, conditionNumber = 0; j < (i * numConditions)
					+ numConditions; j++, conditionNumber++) {
				
				double totalData = 0;
				//this does absolutely nothing since...
				//every number this loop iterates over is the same because for a
				//given set of subjects that belong to a particular condition for
				//a particular group, every subject is the same number.
				//totalData = numSubjects*Constant
				//and we plot totalData / numSubjects = Constant.
				for (int k = j * numSubjects; k < (j * numSubjects) + numSubjects; k++) {
					totalData += designScoresData[k];
				}
				designScoresKeyedData.addValue(conditionNames.get(conditionNumber),
						totalData / numSubjects);
			}

			CategoryDataset designScoresDataset = DatasetUtilities
					.createCategoryDataset("Value", designScoresKeyedData);
			//crashes when i = 1. Number of iterations of i is the number of groups
			String percent = new DecimalFormat("###.###")
					//.format(mPercentage[i][c]);
					.format(mPercentage[0][c]);
			JFreeChart designScoresChart = ChartFactory.createStackedBarChart(
					"Design latent variables for group " + (i + 1)
							+ ", design " + abbrVariableType + " " + (c + 1) + "\n" + percent
							//+ "% crossblock, p < " + mProbability[i][c], "Conditions",
							+ "% crossblock, p < " + mProbability[0][c], "Conditions",
					"Design Scores", designScoresDataset,
					PlotOrientation.VERTICAL, false, true, false);
			
			if (mGlobalButton.isSelected()) {
				applyMinMaxValues(designScoresChart, GLOBAL, RANGE);
			} else if (mLocalButton.isSelected()){
				applyMinMaxValues(designScoresChart, LOCAL, RANGE);
			}
			mDesignScoresChartPanel[i].setChart(designScoresChart);
		}
	}
	
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == mLvComboBox || e.getSource() == mGlobalButton 
				|| e.getSource() == mLocalButton || e.getSource() == mNoneButton) {
			int file = mFileComboBox.getSelectedIndex();
			int lv = mLvComboBox.getSelectedIndex();
			makeLvChart(file, lv);
		} else {
			super.actionPerformed(e);
		}
	}
	
	public void doSaveAs() {
		ChartPanel chartPanel = (ChartPanel) mTabs.getSelectedComponent();
		doSaveAs(chartPanel);
	}
	
	protected boolean ModelIsApplicable(ResultModel model) {
		PlsResultModel plsModel = (PlsResultModel)model;
		return plsModel.getDesignScores() != null && plsModel.getSProbability() != null;
	}

        @Override
	public void notify(FlipVolumeEvent e) {}

	@Override
	public void notify(LoadedVolumesEvent e) {}

	@Override
	public void notify(InvertedLvEvent e) {
                initialize();
	}

	@Override
	public void notify(Event e) {}
}