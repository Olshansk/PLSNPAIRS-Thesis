package pls.chrome.result.view;

import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.event.ActionEvent;
import java.awt.event.AdjustmentEvent;
import java.awt.event.AdjustmentListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.text.DecimalFormat;
import java.util.ArrayList;

import javax.imageio.ImageIO;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollBar;
import javax.swing.JSpinner;
import javax.swing.JTabbedPane;
import javax.swing.JTextField;
import javax.swing.SpinnerNumberModel;
import javax.swing.SpringLayout;
import javax.swing.text.NumberFormatter;

import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.LegendItemCollection;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.category.BarRenderer;
import org.jfree.chart.renderer.xy.StandardXYItemRenderer;
import org.jfree.chart.labels.StandardXYToolTipGenerator;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.statistics.Statistics;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.chart.axis.CategoryAxis;
import org.jfree.chart.axis.NumberAxis;

import pls.shared.GlobalVariablesFunctions;
import pls.chrome.result.DomainAndRangeSetter;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.PlsResultModel;
import pls.chrome.result.model.ResultModel;
import pls.chrome.shared.ImageWriter;
import pls.shared.MLFuncs;

@SuppressWarnings("serial")
public class BrainBehavScoresPlot extends AbstractPlot implements AdjustmentListener {
	
	private JTabbedPane mTabs = new JTabbedPane();
	
	private JComboBox mLvComboBox = new JComboBox();
	private JComboBox mGroupComboBox = new JComboBox();
	private JCheckBox mShowLegendChkBox = new JCheckBox("Show Legend", true);
	
	private JPanel mBrainScoresPanel = new JPanel();
	//private JPanel mCorrelationsPanel = new JPanel();
	private JPanel mBehaviorLvPanel = new JPanel();

	protected ArrayList<ChartPanel> mCorrCharts = new ArrayList<ChartPanel>();
	protected ArrayList<ArrayList<ArrayList<ChartPanel> > > mCharts = new ArrayList<ArrayList<ArrayList<ChartPanel> > >();
	
	private JScrollBar mRowScrollBar = new JScrollBar(JScrollBar.VERTICAL);
	private JScrollBar mColScrollBar = new JScrollBar(JScrollBar.HORIZONTAL);
	
	protected JSpinner mNumRowsTextField = null;
	protected JSpinner mNumColsTextField = null;
	protected SpinnerNumberModel mRowsSpinner = null;
	protected SpinnerNumberModel mColsSpinner = null;
	protected JButton mSetDimsButton = new JButton("Set Dimensions");
	
	private DomainAndRangeSetter mDnrSetter = new DomainAndRangeSetter(0);
	
	private int maxBehavGroupSize;
	private int mNumConditions;
	private int[] mNumSubjectList;
	
	public BrainBehavScoresPlot(String title, GeneralRepository repository) {
		super(title, repository, GlobalVariablesFunctions.PLS);
		
		mFilePanel.add(mLvComboBox);
		mFilePanel.add(mGroupComboBox);
		mFilePanel.add(mShowLegendChkBox);
		mFilePanel.add(mDnrSetter);
		
		remove(mChartPanel);
		add(mTabs, BorderLayout.CENTER);
		
		mBrainScoresPanel.setLayout(new BorderLayout());
		//mCorrelationsPanel.setLayout(new BorderLayout());
		mTabs.add("Brain Scores", mBrainScoresPanel);
		//mTabs.add("Correlation", mCorrelationsPanel);
		
		// Set up the controls for selecting the dimensions of the graphs.
		mRowsSpinner = new SpinnerNumberModel();
		mRowsSpinner.setMinimum(1);
		mRowsSpinner.setStepSize(1);
		
		mColsSpinner = new SpinnerNumberModel();
		mColsSpinner.setMinimum(1);
		mColsSpinner.setStepSize(1);
		
		mNumRowsTextField = new JSpinner(mRowsSpinner);
		mNumColsTextField = new JSpinner(mColsSpinner);
		
		JPanel dimsPanel = new JPanel();
		dimsPanel.setAlignmentX(JPanel.LEFT_ALIGNMENT);
		dimsPanel.setLayout(new BoxLayout(dimsPanel, BoxLayout.Y_AXIS));
		
		JLabel rowLabel = new JLabel("Number of Rows:");
		dimsPanel.add(rowLabel);
		dimsPanel.add(mNumRowsTextField);
		
		JLabel columnLabel = new JLabel("Number of Columns:");
		dimsPanel.add(columnLabel);
		dimsPanel.add(mNumColsTextField);
		
		dimsPanel.add(mSetDimsButton);
		//add(dimsPanel, BorderLayout.WEST);
		
		mSubPane.remove(mButtonPanel);
		mSubPane.remove(mFilePanel);
		
		mSubPane.setLayout(new BorderLayout());
		mSubPane.add(mButtonPanel, BorderLayout.NORTH);
		mSubPane.add(mFilePanel, BorderLayout.CENTER);
		
		mShowLegendChkBox.addActionListener(this);
	
	}
	
	public double[][] getRangeData(ResultModel model) {
		return model.getBrainScores();
	}
	
	public double[][] getDomainData(ResultModel model) {
		return ((PlsResultModel) model).getBehavData();
	}
	
	protected void updateLvComboBoxAndTabs(int fileIndex) {
		PlsResultModel model = (PlsResultModel) mRepository.getGeneral(mResultFilePaths.get(fileIndex) );
		int numLvs = model.getBrainData().getNumLvs();
		String variableType = model.getVariableType();

		mNumSubjectList = model.getNumSubjectList();
		mNumConditions = model.getConditionNames().size();
		
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
		
		// Adds a dropdown menu for selecting a different group.
		if (mGroupComboBox == null) {
			mGroupComboBox = new JComboBox();
		}
		mGroupComboBox.removeActionListener(this);
		mGroupComboBox.removeAllItems();
		
		for (int i = 0; i < mNumSubjectList.length; ++i) {
			mGroupComboBox.addItem("Group #" + (i + 1));
		}
		mGroupComboBox.addActionListener(this);
		
		if (mDnrSetter == null) {
			mDnrSetter = new DomainAndRangeSetter(0);
		}
		
		// Determines the size of the group with the most behav types.
		ArrayList<ArrayList<String>> behavNames = model.getBehavNames();
		maxBehavGroupSize = 0;
		for (int i = 0; i != behavNames.size(); i++) {
			maxBehavGroupSize = Math.max(maxBehavGroupSize, behavNames.get(i).size());
		}
		
		mDnrSetter.setNumPlots(numLvs * mNumSubjectList.length * mNumConditions * maxBehavGroupSize);
		
		if (mShowLegendChkBox == null) {
			mShowLegendChkBox = new JCheckBox("Show Legend", true);
		}
		
		if (mCharts == null) {
			mCharts = new ArrayList<ArrayList<ArrayList<ChartPanel> > >();
		}
		
		if (mCorrCharts == null) {
			mCorrCharts = new ArrayList<ChartPanel>();
		}
		
		if (mBrainScoresPanel == null) {	
			mBrainScoresPanel = new JPanel();
			mBrainScoresPanel.setLayout(new BorderLayout());
			
			mRowScrollBar = new JScrollBar(JScrollBar.VERTICAL);
			mColScrollBar = new JScrollBar(JScrollBar.HORIZONTAL);
		}
		
		//if (mCorrelationsPanel == null) {
		//	mCorrelationsPanel = new JPanel();
		//	mCorrelationsPanel.setLayout(new BorderLayout());
		//}
	}
	
	protected void updateScrollBars(int fileIndex) {
		PlsResultModel model = (PlsResultModel) mRepository.getGeneral(mResultFilePaths.get(fileIndex) );
		
		mRowScrollBar.removeAdjustmentListener(this);
		mColScrollBar.removeAdjustmentListener(this);
		
		int selectedGroup = mGroupComboBox.getSelectedIndex();
		
		mRowScrollBar.setValues(0, 1, 0, model.getConditionNames().size() );
		mColScrollBar.setValues(0, 1, 0, model.getBehavNames().get(selectedGroup).size() );
		
		adjustmentValueChanged(null);
		
		mRowScrollBar.addAdjustmentListener(this);
		mColScrollBar.addAdjustmentListener(this);
	}
	
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == mLvComboBox || e.getSource() == mGlobalButton 
				|| e.getSource() == mLocalButton || e.getSource() == mNoneButton
				|| e.getSource() == mShowLegendChkBox) {
			int file = mFileComboBox.getSelectedIndex();
			int lv = mLvComboBox.getSelectedIndex();
			makeLvChart(file, lv);
		} else if (e.getSource() == mGroupComboBox) {
			int file = mFileComboBox.getSelectedIndex();
			int lv = mLvComboBox.getSelectedIndex();
			makeLvChart(file, lv);
			updateScrollBars(file);
		} else {
			super.actionPerformed(e);
		}
	}
	
	public void adjustmentValueChanged(AdjustmentEvent e) {
		int lv = mLvComboBox.getSelectedIndex();
		int group = mGroupComboBox.getSelectedIndex();
		int behav = mColScrollBar.getValue();
		int cond = mRowScrollBar.getValue();

		mBrainScoresPanel.removeAll();
		
		ChartPanel chartPanel = mCharts.get(group).get(behav).get(cond);
		
		mBrainScoresPanel.add(mColScrollBar, BorderLayout.SOUTH);
		mBrainScoresPanel.add(mRowScrollBar, BorderLayout.EAST);
		mBrainScoresPanel.add(chartPanel, BorderLayout.CENTER);
		
		// Uses the number of groups, the number of conditions and the max number of behav types in a group
		// to calculate a plot index for the domain and range setter. This calculation is done because a
		// 1-dimensional array is used by the domain and range setter.
		updateDomainAndRangeSetter(chartPanel, (lv * mNumSubjectList.length * mNumConditions * maxBehavGroupSize) 
											+ (group * mNumConditions * maxBehavGroupSize)
											+ (behav * mNumConditions)
											+ cond);
		
		if (getParent() != null) {
			getParent().repaint();
		}
	}
	
	public void makeChart(int fileIndex) {
		localRangeMin = Double.POSITIVE_INFINITY;
		localRangeMax = Double.NEGATIVE_INFINITY;
		
		localDomainMin = Double.POSITIVE_INFINITY;
		localDomainMax = Double.NEGATIVE_INFINITY;
		
		// Retrieves the overall min and max values for both the domain data
		// and the range data of the model of the given file index.
		double[][] brainScores = mRangeData.get(fileIndex);
		getMinMaxValues(brainScores, LOCAL, RANGE);
		
		double[][] behavData = mDomainData.get(fileIndex);
		getMinMaxValues(behavData, LOCAL, DOMAIN);
		
		// Uses the first LV as the default selected LV.
		makeLvChart(fileIndex, 0);
	}
	
	public void makeLvChart(int fileIndex, int c) {
		mCharts.clear();
		mCorrCharts.clear();
		
		PlsResultModel model = (PlsResultModel) mRepository.getGeneral(mResultFilePaths.get(fileIndex) );
		double[][] brainScores = mRangeData.get(fileIndex);
		double[][] behavData = mDomainData.get(fileIndex);
		
		ArrayList<String> conditionNames = model.getConditionNames();
		String abbrVariableType = model.getAbbrVariableType().toUpperCase();
		
		ArrayList<ArrayList<String>> subjectNames = model.getSubjectNames();
		ArrayList<ArrayList<String>> behavNames = model.getBehavNames();
		
		makeBehavScoresPlot(brainScores, behavData, conditionNames,
				abbrVariableType, subjectNames, behavNames, c);
		
		if (getParent() != null) {
			getParent().repaint();
		}
	}
	
	private void makeBehavScoresPlot(double[][] brainScores, double[][] behavData,
			ArrayList<String> conditionNames, String abbrVariableType,
			ArrayList<ArrayList<String>> subjectNames,
			ArrayList<ArrayList<String>> behavNames, int c) {
		boolean b = mShowLegendChkBox.isSelected();

		double[] bScr = MLFuncs.getColumn(brainScores, c);

		ArrayList<ArrayList<ArrayList<ArrayList<XYSeries> > > > someSeries = new ArrayList<ArrayList<ArrayList<ArrayList<XYSeries> > > >();

		StandardXYToolTipGenerator tooltip = new StandardXYToolTipGenerator();
		StandardXYItemRenderer renderer = new StandardXYItemRenderer(
				StandardXYItemRenderer.SHAPES, tooltip);

		// i = group number
		for (int i = 0; i < mNumSubjectList.length; i++) {
			
			ArrayList<ArrayList<ArrayList<XYSeries> > > group = new ArrayList<ArrayList<ArrayList<XYSeries> > >();
			someSeries.add(group);
			int numSubjects = mNumSubjectList[i];
			
			for (int behavNum = 0; behavNum < behavNames.get(i).size(); ++behavNum) {
				
				double[] bData = MLFuncs.getColumn(behavData, behavNum);
				ArrayList<ArrayList<XYSeries> > behav = new ArrayList<ArrayList<XYSeries> >();
				group.add(behav);
				
				for (int j = i * mNumConditions, conditionNumber = 0; j < (i * mNumConditions)
						+ mNumConditions; j++, conditionNumber++) {
					
					for (int k = j * numSubjects, subjectNumber = 0; k < (j * numSubjects) 
						+ numSubjects; k++, subjectNumber++) {
						
						if (conditionNumber >= behav.size() ) {
							behav.add(new ArrayList<XYSeries>() );
						}
						XYSeries series = null;
						series = new XYSeries("Subject: " + subjectNames.get(i).get(subjectNumber) );
						behav.get(conditionNumber).add(series);
						
						series.add(bData[k], bScr[k]);
					}
				}
			}
		}

		// Loop through each group
		for (int g = 0; g < someSeries.size(); ++g) {
			ArrayList<ArrayList<ArrayList<XYSeries> > > group = someSeries.get(g);
			ArrayList<ArrayList<ChartPanel> > groupPanel = new ArrayList<ArrayList<ChartPanel> >();
			
			DefaultCategoryDataset catdata = new DefaultCategoryDataset();
			
			// Loop through each behaviour
			for (int beh = 0; beh < group.size(); ++beh) {
				
				ArrayList<ArrayList<XYSeries> > behav = group.get(beh);
				ArrayList<ChartPanel> behavPanel = new ArrayList<ChartPanel>();
			
				// For each group of conditions, make a new plot
				for (int i = 0; i < behav.size(); ++i) {
					NumberAxis domainAxis = new NumberAxis(behavNames.get(g).get(beh) );
					domainAxis.setAutoRangeIncludesZero(false);
					NumberAxis rangeAxis = new NumberAxis(conditionNames.get(i));
					rangeAxis.setAutoRangeIncludesZero(false);
					
					XYSeriesCollection dataset = new XYSeriesCollection();
					
					ArrayList<Number> xVals = new ArrayList<Number>();
					ArrayList<Number> yVals = new ArrayList<Number>();
					
					ArrayList<XYSeries> subjects = behav.get(i);
					
					for (XYSeries s : subjects) {
						xVals.add(s.getX(0) );
						yVals.add(s.getY(0) );
						dataset.addSeries(s);
					}
					
					Number[] xArray = new Number[xVals.size()];
					Number[] yArray = new Number[yVals.size()];
					
					xVals.toArray(xArray);
					yVals.toArray(yArray);
					
					Double r = Statistics.getCorrelation(xArray, yArray);
					
					catdata.addValue(r, conditionNames.get(i), behavNames.get(g).get(beh));
					
					String rString = Double.toString(r);
					
					NumberFormatter numFormatter = new NumberFormatter(new DecimalFormat("0.###"));
					try {
						rString = numFormatter.valueToString(r);
					}
					catch (Exception e) {
						e.printStackTrace();
					}
					
					XYPlot plot = new XYPlot(dataset, domainAxis, rangeAxis, renderer);
					
	
					JFreeChart chart = new JFreeChart("Brain vs. Behaviour Scores, "
							+ abbrVariableType + " " + (c + 1) + ", r = " + rString, plot);
					
					if (!b) {
						plot.setFixedLegendItems(new LegendItemCollection() );
					}
					
					if (mGlobalButton.isSelected()) {
						applyMinMaxValues(chart, GLOBAL, RANGE);
						applyMinMaxValues(chart, GLOBAL, DOMAIN);
					} else if (mLocalButton.isSelected()){
						applyMinMaxValues(chart, LOCAL, RANGE);
						applyMinMaxValues(chart, LOCAL, DOMAIN);
					}

					behavPanel.add(new ChartPanel(chart));
				}
				groupPanel.add(behavPanel);
			}
			
			NumberAxis rangeAxis = new NumberAxis("Correlation");
			rangeAxis.setRange(-1, 1);
			CategoryPlot catplot = new CategoryPlot(catdata, new CategoryAxis("Behavior"), rangeAxis, new BarRenderer());
//			CategoryPlot catplot = new CategoryPlot();
//			catplot.setDataset(catdata);
			
			JFreeChart chart = new JFreeChart("Correlations", catplot);
			
			mCharts.add(groupPanel);
			mCorrCharts.add(new ChartPanel(chart) );
		}

		int selectedGroup = mGroupComboBox.getSelectedIndex();
		int selectedBehav = mColScrollBar.getValue();
		int selectedCondition = mRowScrollBar.getValue();
		
		ChartPanel chartPanel = mCharts.get(selectedGroup).get(selectedBehav).get(selectedCondition);
		
		// Uses the number of groups, the number of conditions and the max number of behav types in a group
		// to calculate a plot index for the domain and range setter. This calculation is done because a
		// 1-dimensional array is used by the domain and range setter.
		updateDomainAndRangeSetter(chartPanel, (c * mNumSubjectList.length * mNumConditions * maxBehavGroupSize) 
											+ (selectedGroup * mNumConditions * maxBehavGroupSize)
											+ (selectedBehav * mNumConditions)
											+ selectedCondition);
		
		// Removes the old charts that are currently being displayed first such that
		// there are no overlaps.
		if (mBrainScoresPanel.getComponentCount() != 0) {
			mBrainScoresPanel.remove(mBrainScoresPanel.getComponentCount() - 1);
		}
		mBrainScoresPanel.add(chartPanel, BorderLayout.CENTER);
		
		//mCorrelationsPanel.removeAll();
		//mCorrelationsPanel.add(mCorrCharts.get(selectedGroup), BorderLayout.CENTER);
	}
	
	private void updateDomainAndRangeSetter(ChartPanel chartPanel, int plotIndex) {
		XYPlot plot = chartPanel.getChart().getXYPlot();
		
		mDnrSetter.setAxes(plot.getDomainAxis(), plot.getRangeAxis(), plotIndex);
		mDnrSetter.updateInputFields();
	}
	
	public void doSaveAs() {
		Component component = mTabs.getSelectedComponent();
		//if (component == mCorrelationsPanel) {
		//	int group = mGroupComboBox.getSelectedIndex();
		//	doSaveAs(mCorrCharts.get(group));
		//} else {
			saveBrainScores();
		//}
	}
	
	private void saveBrainScores() {
		int group = mGroupComboBox.getSelectedIndex();
		int behav = mColScrollBar.getValue();
		int cond = mRowScrollBar.getValue();
		ImageWriter.saveBBScoresPlot(this, group, behav, cond, mCharts);
	}
	
	protected boolean ModelIsApplicable(ResultModel model) {
		PlsResultModel plsModel = (PlsResultModel)model;
		return plsModel != null && plsModel.getBrainScores() != null && plsModel.getBehavData() != null;
	}
}
