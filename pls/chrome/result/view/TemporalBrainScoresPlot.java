package pls.chrome.result.view;

import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Vector;

import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SpringLayout;

import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.labels.StandardXYToolTipGenerator;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.StandardXYItemRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import pls.analysis.ConcatenateFmriDatamat;
import pls.chrome.result.controller.observer.Event;
import pls.chrome.result.controller.observer.datachange.DataChangeObserver;
import pls.chrome.result.controller.observer.datachange.FlipVolumeEvent;
import pls.chrome.result.controller.observer.datachange.InvertedLvEvent;
import pls.chrome.result.controller.observer.datachange.LoadedVolumesEvent;
import pls.chrome.result.model.BrainData;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.PlsResultModel;
import pls.chrome.result.model.ResultModel;
import pls.shared.GlobalVariablesFunctions;
import pls.shared.MLFuncs;
import Jama.Matrix;

@SuppressWarnings("serial")
public class TemporalBrainScoresPlot extends AbstractFunctionalPlot 
		implements DataChangeObserver{
	
	private JLabel mLvLabel = new JLabel();
	private JComboBox mLvComboBox = new JComboBox();
	private ArrayList<String[]> mSessionProfiles;
	private String mCurrentDirectory;
	private int[] mConditionSelection;
	
	protected ArrayList<HashMap<Integer, double[]>> mBrainLv;
	protected ArrayList<Integer> mCoords;
	protected double[][][] mDatamat;
	
	protected ConcatenateFmriDatamat st;
	
	protected String mPlotType;
	
	public TemporalBrainScoresPlot(String title, GeneralRepository repository) {
		super(title, repository, GlobalVariablesFunctions.PLS, false);
		mPlotType = "Brain Scores";
		
		initializeWidgets();
		mRepository.getPublisher().registerObserver(this);
	}
	
	public TemporalBrainScoresPlot(String title, GeneralRepository repository, boolean behav) {
		super(title, repository, GlobalVariablesFunctions.PLS, behav);
		mPlotType = "Brain Scores";
		
		initializeWidgets();
	}
	
	private void initializeWidgets() {
		mProgressBar.setMinimum(0);
		mProgressBar.setMaximum(14);
		
		mSetDimsButton.setText("PLOT");
		
		JPanel lvPanel = new JPanel();
		lvPanel.setAlignmentX(JPanel.LEFT_ALIGNMENT);
		SpringLayout s = new SpringLayout();
		lvPanel.setLayout(s);
		
		lvPanel.add(mLvLabel);
		lvPanel.add(mLvComboBox);

		int width = mGroupComboBox.getPreferredSize().width;
		int height = mLvComboBox.getPreferredSize().height;
		mLvComboBox.setPreferredSize(new Dimension(width, height));
		
		s.putConstraint(SpringLayout.NORTH, mLvLabel, 30, SpringLayout.NORTH, lvPanel);
		s.putConstraint(SpringLayout.WEST, mLvLabel, 5, SpringLayout.WEST, lvPanel);
		s.putConstraint(SpringLayout.NORTH, mLvComboBox, 0, SpringLayout.SOUTH, mLvLabel);
		s.putConstraint(SpringLayout.WEST, mLvComboBox, 5, SpringLayout.WEST, lvPanel);
		
		width = mLvComboBox.getPreferredSize().width + 5;
		height = mLvLabel.getPreferredSize().height
				   + mLvComboBox.getPreferredSize().height
				   + 50;
		Dimension dimension = new Dimension(width, height);
		lvPanel.setPreferredSize(dimension);
		lvPanel.setMaximumSize(dimension);
		lvPanel.setMinimumSize(dimension);
		
		mLeftPanel.add(mProgressPanel);
		mLeftPanel.add(lvPanel, 2);
	}
	
	protected void updateLvComboBoxAndTabs(int fileIndex) {
		super.updateLvComboBoxAndTabs(fileIndex);
		PlsResultModel model = mRepository.getPlsModel(mResultFilePaths.get(fileIndex));
		
		int numLvs = model.getBrainData().getNumLvs();
		String abbrVariableType = model.getAbbrVariableType().toUpperCase();
		
		if (mLvLabel == null) {
			mLvLabel = new JLabel(abbrVariableType + ":");
		} else {
			mLvLabel.setText(abbrVariableType + ":");
		}

		if (mLvComboBox == null) {
			mLvComboBox = new JComboBox();
		} else {
			mLvComboBox.removeActionListener(this);
			mLvComboBox.removeAllItems();
		}
		for (int i = 0; i < numLvs; i++) {
			mLvComboBox.addItem(abbrVariableType + " #" + (i + 1) );
		}
		mLvComboBox.setPreferredSize(new Dimension(100,
				mLvComboBox.getPreferredSize().height));
//		mLvComboBox.setPreferredSize(new Dimension(100,
//				15));
		mLvComboBox.addActionListener(this);
		
		mSessionProfiles = model.getSessionProfileArray();
		mCurrentDirectory = model.getFileDir();
		mConditionSelection = model.getConditionSelection();
		mBrainLv = model.getBrainData(BrainData.BRAINLV_STRING).getAllData();
		mCoords = new ArrayList<Integer>(model.getFilteredCoordinates() );

		if (mSetDimsButton == null) {
			mSetDimsButton = new JButton("PLOT");
		} else {
			mSetDimsButton.setText("PLOT");
		}
		
		if (mChartsPanel == null) {
			mChartsPanel = new JPanel();
		}
		mChartsPanel.setVisible(false);
	}
	
	protected Vector<Vector<double[]>> calculateData(int group, int lv) {
		int winSize = st.winSize;
		int numConditions = st.conditions.length;
		int numVoxels = st.numVoxels;
		int numEvents = st.eventList.length;
		
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
		for (int i = 0; i != winSize; i++) {
			Matrix datamatMatrix = new Matrix(MLFuncs.getXYColumns(mDatamat, i));
			Matrix blvMatrix = new Matrix(blv[i]);
			brainScores[i] = datamatMatrix.times(blvMatrix).getArray();
		}
		
		Vector<Vector<double[]>> result = new Vector<Vector<double[]>>();
		
		int currSubject = 0;
		for (int i = 0; i != group; i++) {
			currSubject += numConditions * st.numSubjectList[i];
		}
		
		for (int j = 0; j < numConditions; j++) {
			Vector<double[]> currConditionResult = new Vector<double[]>();
			double[][] thisConditionValues = new double[st.numSubjectList[group]][winSize];
			
			for (int k = 0; k < st.numSubjectList[group]; k++) {
				double[] plotData = new double[winSize];
				for (int m = 0; m != winSize; m++) {
					plotData[m] = brainScores[m][currSubject][0];
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
	
	public void initialize() {
		super.initialize();
		
		if (st != null) {
			initializeCharts();
		}
	}
	
	protected void createCharts() {
		int c = mLvComboBox.getSelectedIndex();
		int i = mGroupComboBox.getSelectedIndex();
		
		int numSubjects = getNumSubjects(i);
		
		mCharts = new ChartPanel[st.conditions.length][numSubjects];
		if (numSubjects == 1) {
			mAverages = null;
			mShowAverageCheckBox.setVisible(false);
		} else {
			mAverages = new ChartPanel[st.conditions.length];
			mShowAverageCheckBox.setVisible(true);
		}

		Vector<Vector<double[]>> currGroupLvResult = calculateData(i, c);

		for (int j = 0; j < st.conditions.length; j++) {
			Vector<double[]> currConditionResult = currGroupLvResult.get(j);

			for (int k = 0; k < numSubjects + 1; k++) {
				if (k == numSubjects && numSubjects == 1) {
					break;
				}
				XYSeriesCollection dataset = new XYSeriesCollection();
				StandardXYToolTipGenerator tooltip = new StandardXYToolTipGenerator();
				StandardXYItemRenderer renderer = new StandardXYItemRenderer(
						StandardXYItemRenderer.SHAPES_AND_LINES, tooltip);

				double[] currSubjectResult = currConditionResult.get(k);

				XYSeries series = null;
				if (k == numSubjects) {
					series = new XYSeries("Mean for condition: "
							+ st.conditions[j]);
				} else {
					series = new XYSeries("LV " + (c + 1) + ", Group: "
							+ (i + 1) + ", Condition: " + st.conditions[j]
							+ ", Subject: " + (k + 1));
				}

				for (int l = 0; l < currSubjectResult.length; l++) {
					series.add(l, currSubjectResult[l]);
				}

				dataset.addSeries(series);

				NumberAxis domainAxis = new NumberAxis("Lag");
				domainAxis.setAutoRangeIncludesZero(false);
				NumberAxis rangeAxis = new NumberAxis(mPlotType);
				rangeAxis.setAutoRangeIncludesZero(false);

				XYPlot plot = new XYPlot(dataset, domainAxis, rangeAxis,
						renderer);

				JFreeChart chart = null;
				if (k == numSubjects) {
					chart = new JFreeChart("Temporal " + mPlotType + " Plot, "
							+ "Mean for condition: " + st.conditions[j],
							plot);
				} else {
					chart = new JFreeChart("Temporal " + mPlotType + " Plot, "
							+ "LV " + (c + 1) + ", Group: " + (i + 1)
							+ ", Condition: " + st.conditions[j]
							+ ", Subject: " + (k + 1), plot);
				}
				chart.removeLegend();
				
				if (k == numSubjects) {
					mAverages[j] = new ChartPanel(chart);
				} else {
					mCharts[j][k] = new ChartPanel(chart);
				}
			}
		}
		
		if (mProgressBar != null) {
			mProgressBar.setValue(mProgressBar.getValue() + 2);
		}
	}
	
	protected int getNumSubjects(int group) {
		return st.numSubjectList[group];
	}
	
	public void actionPerformed(final ActionEvent event) {
		if ((event.getSource() == mSetDimsButton && mSetDimsButton.getText().equals("PLOT"))) {
			
			// Generates the datamat on a separate thread such that the progess
			// bar used by it can be displayed properly.
			mProgressBar.setValue(0);
			new Thread(){
				public void run() {
					mSetDimsButton.setEnabled(false);
					mSetDimsButton.setText("Please wait...");
					try {
						st = new ConcatenateFmriDatamat(new Vector<String[]>(mSessionProfiles), mCurrentDirectory, mConditionSelection, mProgressBar);
						mDatamat = MLFuncs.reshape(st.datamat, st.eventList.length, st.winSize, st.numVoxels);
					} catch (Exception e) {
						String message = e.getMessage();
						if (message != null) {
							GlobalVariablesFunctions.showErrorMessage("Temporal " + mPlotType + " could not be plotted:\n" + message);
						} else {
							GlobalVariablesFunctions.showErrorMessage("Temporal " + mPlotType + " could not be plotted.");
						}
						e.printStackTrace();
						return;
					}
					initializeCharts();
					superActionPerformed(event);
					
					mSetDimsButton.setText("Set Dimensions");
					mSetDimsButton.setEnabled(true);
					mChartsPanel.setVisible(true);
				}
			}.start();
		} else if ( event.getSource() == mGroupComboBox || event.getSource() == mLvComboBox ) {

			//Don't do anything if we have never plotted anything in the first place.
			if(mSetDimsButton.getText().equals("PLOT"))
				return;

			// don't redo EVERYTHING if we're just changing the LV/group, just
			// recreate the charts and adjust the dnr
			createCharts();
			adjustDomainAndRange();
			if (mProgressBar != null) {
				mProgressBar.setValue(mProgressBar.getMaximum());
			}
			
			super.actionPerformed(event);
		} else {
			//st = null;
			super.actionPerformed(event);
		}
	}
	
	// This method was written such that it can be called from
	// inside a new thread.
	private void superActionPerformed(ActionEvent e) {
		super.actionPerformed(e);
	}
	
	protected boolean ModelIsApplicable(ResultModel model) {
		String contrastFile = model.getConstrastFilename();
		boolean sessionEmpty = false;

		if(model.getSessionProfiles() == null)
			sessionEmpty = true;

		return model.getWindowSize() > 1 &&
				contrastFile != null &&
				(((PlsResultModel)model).getBehavData() == null) && 
				!sessionEmpty;
	}

	@Override
	public void notify(FlipVolumeEvent e) {}

	@Override
	public void notify(LoadedVolumesEvent e) {}

	@Override
	public void notify(InvertedLvEvent e) {
		//Simulate an mGroupComboBox/mLvComboBox event
		//so the charts and dnr are redrawn.
			 
		mLvComboBox.setSelectedIndex(mLvComboBox.getSelectedIndex());
	}

	@Override
	public void notify(Event e) {}
}