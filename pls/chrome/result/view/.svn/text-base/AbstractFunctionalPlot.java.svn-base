package pls.chrome.result.view;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.AdjustmentEvent;
import java.awt.event.AdjustmentListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;

import javax.imageio.ImageIO;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JScrollBar;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;
import javax.swing.SpringLayout;

import javax.swing.text.JTextComponent;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.XYPlot;

import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.PlsResultModel;
import pls.chrome.result.model.ResultModel;
import pls.chrome.shared.ImageWriter;
import pls.shared.GlobalVariablesFunctions;

@SuppressWarnings("serial")
public abstract class AbstractFunctionalPlot extends AbstractPlot implements AdjustmentListener {
	
	private int mNumRows = 0;
	private int mNumCols = 0;
	
	private boolean mShowAverage = true;
	
	protected String mCurrentFilePath;
	
	protected JComboBox mGroupComboBox = new JComboBox();
	protected JSpinner mNumRowsTextField = null;
	protected JSpinner mNumColsTextField = null;
	protected SpinnerNumberModel mRowsSpinner = new SpinnerNumberModel();
	protected SpinnerNumberModel mColsSpinner = new SpinnerNumberModel();
	protected JButton mSetDimsButton = new JButton("Set Dimensions");
	protected JCheckBox mShowAverageCheckBox = new JCheckBox("Show Average Plots", false);
	protected ChartPanel[][] mCharts = null;
	protected ChartPanel[] mAverages = null;
	protected JPanel mLeftPanel = new JPanel();
	
	protected JPanel mChartsPanel = new JPanel();
	protected JPanel mChartsInnerPanel = new JPanel();
	protected JScrollBar mRowScrollBar = new JScrollBar(JScrollBar.VERTICAL);
	protected JScrollBar mColScrollBar = new JScrollBar(JScrollBar.HORIZONTAL);
	protected ArrayList<ArrayList<String>> mXAxisNames = null;
	protected ArrayList<String> mYAxisNames = null;
	
	protected JPanel mProgressPanel = null;
	protected JProgressBar mProgressBar = new JProgressBar(JProgressBar.HORIZONTAL);
	
	protected boolean mBehavPlot;
	
	public AbstractFunctionalPlot(String title, GeneralRepository repository, String fileType, boolean behavPlot) {
		super(title, repository, fileType);
		mBehavPlot = behavPlot;
		
		// This plot is not using the layout that was setup in AbstractPlot,
		// but its widgets are still being used though.
		removeAll();
		
		setupWidgets();
	}
	
	private void setupWidgets() {
		setLayout(new BoxLayout(this, BoxLayout.X_AXIS));
		
		mRowsSpinner.setMinimum(1);
		mColsSpinner.setMinimum(1);
		mNumRowsTextField = new JSpinner(mRowsSpinner);
		mNumColsTextField = new JSpinner(mColsSpinner);
		
		// Set up the button for attaching/detaching from the main
		// results displayer.
		JPanel buttonPanel = new JPanel();
		buttonPanel.setAlignmentX(JPanel.LEFT_ALIGNMENT);
		SpringLayout s = new SpringLayout();
		buttonPanel.setLayout(s);
		buttonPanel.add(mAttachDetachButton);
		
		s.putConstraint(SpringLayout.NORTH, mAttachDetachButton, 0, SpringLayout.NORTH, buttonPanel);
		s.putConstraint(SpringLayout.WEST, mAttachDetachButton, 5, SpringLayout.WEST, buttonPanel);

		int width = mAttachDetachButton.getPreferredSize().width + 5;
		int height = mAttachDetachButton.getPreferredSize().height;
		Dimension dimension = new Dimension(width, height);
		buttonPanel.setPreferredSize(dimension);
		buttonPanel.setMaximumSize(dimension);
		buttonPanel.setMinimumSize(dimension);
		
		// Set up the controls for selecting a result file.
		JPanel filePanel = new JPanel();
		filePanel.setAlignmentX(JPanel.LEFT_ALIGNMENT);
		s = new SpringLayout();
		filePanel.setLayout(s);
		
		JLabel fileLabel = new JLabel("File:");
		filePanel.add(fileLabel);
		filePanel.add(mFileComboBox);
		
		s.putConstraint(SpringLayout.NORTH, fileLabel, 30, SpringLayout.NORTH, filePanel);
		s.putConstraint(SpringLayout.WEST, fileLabel, 5, SpringLayout.WEST, filePanel);
		s.putConstraint(SpringLayout.NORTH, mFileComboBox, 0, SpringLayout.SOUTH, fileLabel);
		s.putConstraint(SpringLayout.WEST, mFileComboBox, 5, SpringLayout.WEST, filePanel);

		width = mFileComboBox.getPreferredSize().width + 5;
		height = fileLabel.getPreferredSize().height
			   + mFileComboBox.getPreferredSize().height + 30;
		dimension = new Dimension(width, height);
		filePanel.setPreferredSize(dimension);
		filePanel.setMaximumSize(dimension);
		filePanel.setMinimumSize(dimension);
		
		// Set up the controls for selecting a group.
		JPanel groupPanel = new JPanel();
		groupPanel.setAlignmentX(JPanel.LEFT_ALIGNMENT);
		s = new SpringLayout();
		groupPanel.setLayout(s);
		
		JLabel groupLabel = new JLabel("Group:");
		groupPanel.add(groupLabel);
		groupPanel.add(mGroupComboBox);

		width = mGroupComboBox.getPreferredSize().width + 100;
		height = mGroupComboBox.getPreferredSize().height;
		mGroupComboBox.setPreferredSize(new Dimension(width, height));
		
		s.putConstraint(SpringLayout.NORTH, groupLabel, 30, SpringLayout.NORTH, groupPanel);
		s.putConstraint(SpringLayout.WEST, groupLabel, 5, SpringLayout.WEST, groupPanel);
		s.putConstraint(SpringLayout.NORTH, mGroupComboBox, 0, SpringLayout.SOUTH, groupLabel);
		s.putConstraint(SpringLayout.WEST, mGroupComboBox, 5, SpringLayout.WEST, groupPanel);
		
		width = mGroupComboBox.getPreferredSize().width + 5;
		height = groupLabel.getPreferredSize().height
			   + mGroupComboBox.getPreferredSize().height + 30;
		dimension = new Dimension(width, height);
		groupPanel.setPreferredSize(dimension);
		groupPanel.setMaximumSize(dimension);
		groupPanel.setMinimumSize(dimension);
		
		// Set up the controls for selecting the dimensions of the graphs.
		JPanel dimsPanel = new JPanel();
		dimsPanel.setAlignmentX(JPanel.LEFT_ALIGNMENT);
		s = new SpringLayout();
		dimsPanel.setLayout(s);
		
		JLabel rowLabel = new JLabel("Number of Rows:");
		dimsPanel.add(rowLabel);
		dimsPanel.add(mNumRowsTextField);
		
		JLabel columnLabel = new JLabel("Number of Columns:");
		dimsPanel.add(columnLabel);
		dimsPanel.add(mNumColsTextField);
		
		dimsPanel.add(mSetDimsButton);
		dimsPanel.add(mShowAverageCheckBox);
		
		s.putConstraint(SpringLayout.NORTH, rowLabel, 40, SpringLayout.NORTH, dimsPanel);
		s.putConstraint(SpringLayout.WEST, rowLabel, 5, SpringLayout.WEST, dimsPanel);
		s.putConstraint(SpringLayout.NORTH, mNumRowsTextField, 0, SpringLayout.SOUTH, rowLabel);
		s.putConstraint(SpringLayout.WEST, mNumRowsTextField, 5, SpringLayout.WEST, dimsPanel);
		
		s.putConstraint(SpringLayout.NORTH, columnLabel, 10, SpringLayout.SOUTH, mNumRowsTextField);
		s.putConstraint(SpringLayout.WEST, columnLabel, 5, SpringLayout.WEST, dimsPanel);
		s.putConstraint(SpringLayout.NORTH, mNumColsTextField, 0, SpringLayout.SOUTH, columnLabel);
		s.putConstraint(SpringLayout.WEST, mNumColsTextField, 5, SpringLayout.WEST, dimsPanel);
		
		s.putConstraint(SpringLayout.NORTH, mSetDimsButton, 10, SpringLayout.SOUTH, mNumColsTextField);
		s.putConstraint(SpringLayout.WEST, mSetDimsButton, 5, SpringLayout.WEST, dimsPanel);
		s.putConstraint(SpringLayout.NORTH, mShowAverageCheckBox, 0, SpringLayout.SOUTH, mSetDimsButton);
		s.putConstraint(SpringLayout.WEST, mShowAverageCheckBox, 5, SpringLayout.WEST, dimsPanel);
		
		width = mShowAverageCheckBox.getPreferredSize().width + 5;
		height = rowLabel.getPreferredSize().height
				   + mNumRowsTextField.getPreferredSize().height
				   + columnLabel.getPreferredSize().height
				   + mNumColsTextField.getPreferredSize().height
				   + mSetDimsButton.getPreferredSize().height
				   + mShowAverageCheckBox.getPreferredSize().height
				   + 60;
		dimension = new Dimension(width, height);
		dimsPanel.setPreferredSize(dimension);
		dimsPanel.setMaximumSize(dimension);
		dimsPanel.setMinimumSize(dimension);
		
		// Set up the progress bar.
		mProgressPanel = new JPanel();
		mProgressPanel.setAlignmentX(JPanel.LEFT_ALIGNMENT);
		s = new SpringLayout();
		mProgressPanel.setLayout(s);
		mProgressPanel.add(mProgressBar);
		
		s.putConstraint(SpringLayout.NORTH, mProgressBar, 0, SpringLayout.NORTH, mProgressPanel);
		s.putConstraint(SpringLayout.WEST, mProgressBar, 5, SpringLayout.WEST, mProgressPanel);

		width = mProgressBar.getPreferredSize().width + 5;
		height = mProgressBar.getPreferredSize().height;
		dimension = new Dimension(width, height);
		mProgressPanel.setPreferredSize(dimension);
		mProgressPanel.setMaximumSize(dimension);
		mProgressPanel.setMinimumSize(dimension);
		
		mLeftPanel.setLayout(new BoxLayout(mLeftPanel, BoxLayout.Y_AXIS));
		mLeftPanel.setAlignmentY(JPanel.TOP_ALIGNMENT);
		mLeftPanel.add(buttonPanel);
		mLeftPanel.add(filePanel);
		mLeftPanel.add(groupPanel);
		mLeftPanel.add(dimsPanel);
		add(mLeftPanel);
		
		mSetDimsButton.addActionListener(this);
		mShowAverageCheckBox.addActionListener(this);
		
		mChartsPanel.setLayout(new BorderLayout());
		mChartsPanel.setAlignmentY(JPanel.TOP_ALIGNMENT);
		
		add(mChartsPanel);
		
		mRowScrollBar.addAdjustmentListener(this);
		mColScrollBar.addAdjustmentListener(this);
	}
	
	protected void updateLvComboBoxAndTabs(int fileIndex) {
		PlsResultModel model = mRepository.getPlsModel(mResultFilePaths.get(fileIndex));
		
		if (mBehavPlot) {
			mXAxisNames = model.getBehavNames();
		} else {
			mXAxisNames = model.getSubjectNames();
		}
		mYAxisNames = model.getConditionNames();
		
		if (mRowsSpinner == null) {
			mRowsSpinner = new SpinnerNumberModel();
			mRowsSpinner.setMinimum(1);
			mRowsSpinner.setStepSize(1);
			
			mColsSpinner = new SpinnerNumberModel();
			mColsSpinner.setMinimum(1);
			mColsSpinner.setStepSize(1);
		}
		mRowsSpinner.setValue(Math.min(4, mYAxisNames.size()));
		mRowsSpinner.setMaximum(mYAxisNames.size());
		
		mColsSpinner.setValue(Math.min(2, mXAxisNames.get(0).size()));
		mColsSpinner.setMaximum(mXAxisNames.get(0).size());
		
		if (mGroupComboBox == null) {
			mGroupComboBox = new JComboBox();
		} else {
			mGroupComboBox.removeActionListener(this);
			mGroupComboBox.removeAllItems();
		}
		int numGroups = model.getSessionProfileArray().size();
		for (int i = 1; i <= numGroups; ++i) {
			mGroupComboBox.addItem("Group " + i);
		}
		mGroupComboBox.addActionListener(this);
	}
	
	// This function is returning null since the min and max range values
	// are not required here.
	public double[][] getRangeData(ResultModel model) {
		return null;
	}
	
	// This function is not being used here since the charts are not
	// being automatically generated whenever a user switches result files.
	public void makeChart(int fileIndex) {}
	
	protected void initializeCharts() {
		if (mResultFilePaths.size() > 0) {
			createCharts();
			adjustDomainAndRange();
			if (mProgressBar != null) {
				mProgressBar.setValue(mProgressBar.getMaximum());
			}
		}
	}
	
	abstract protected void createCharts();
	
	protected void adjustDomainAndRange() {
		
		Double max = Double.MIN_VALUE;
		Double min = Double.MAX_VALUE;
		
		JFreeChart chart;
		XYPlot plot;
		ValueAxis rangeAxis;
		for (int i = 0; i < mCharts.length; i++) {
			
			for (int j = 0; j != mCharts[i].length; j++) {
				chart = mCharts[i][j].getChart();
				plot = chart.getXYPlot();
				
				rangeAxis = plot.getRangeAxis();
				rangeAxis.setAutoRange(false);
				
				if (min > rangeAxis.getLowerBound() ) {
					min = rangeAxis.getLowerBound();
				}
				
				if (max < rangeAxis.getUpperBound() ) {
					max = rangeAxis.getUpperBound();
				}
			}
			
			if (mAverages != null) {
				chart = mAverages[i].getChart();
				plot = chart.getXYPlot();
				
				rangeAxis = plot.getRangeAxis();
				rangeAxis.setAutoRange(false);
				
				if (min > rangeAxis.getLowerBound() ) {
					min = rangeAxis.getLowerBound();
				}
				
				if (max < rangeAxis.getUpperBound() ) {
					max = rangeAxis.getUpperBound();
				}
			}
		}
		
		for (int i = 0; i < mCharts.length; ++i) {
			for (ChartPanel cp : mCharts[i]) {
				chart = cp.getChart();
				plot = chart.getXYPlot();
				
				rangeAxis = plot.getRangeAxis();
				rangeAxis.setAutoRange(false);
				
				rangeAxis.setLowerBound(min);
				rangeAxis.setUpperBound(max);
			}

			if (mAverages != null) {
				chart = mAverages[i].getChart();
				plot = chart.getXYPlot();
				
				rangeAxis = plot.getRangeAxis();
				rangeAxis.setAutoRange(false);
				
				rangeAxis.setLowerBound(min);
				rangeAxis.setUpperBound(max);
			}
		}
	}
	
	protected void rearrangePlots() {
		if (mCharts == null) {
			return;
		}
		
		mChartsPanel.removeAll();
		layoutPlots(0, 0);
		
		mChartsPanel.add(mChartsInnerPanel, BorderLayout.CENTER);
		mRowScrollBar.setValues(0, mNumRows, 0, mCharts.length);
		mColScrollBar.setValues(0, mNumCols, 0, mCharts[0].length);
		mChartsPanel.add(mRowScrollBar, BorderLayout.EAST);
		mChartsPanel.add(mColScrollBar, BorderLayout.SOUTH);
	}
	
	private void layoutPlots(int rindex, int cindex) {
		int group = mGroupComboBox.getSelectedIndex();
		
		mChartsInnerPanel.removeAll();
		
		int cols = mNumCols;
		if (mShowAverage) {
			++cols;
		}
		
		GridLayout gl = new GridLayout(mNumRows, cols);
		mChartsInnerPanel.setLayout(gl);
		
		for (int i = rindex; i < rindex + mNumRows; ++i) {
			for (int j = cindex; j < cindex + mNumCols; ++j) {
				ChartPanel cp = mCharts[i][j];
				JFreeChart jfc = cp.getChart();
				
				if (i == rindex) {
					jfc.setTitle(mXAxisNames.get(group).get(j) );
				}
				else {
					jfc.setTitle("");
				}
				
				XYPlot xyplot = jfc.getXYPlot();
				xyplot.getRangeAxis().setVisible(j == cindex);
				xyplot.getRangeAxis().setLabel(mYAxisNames.get(i) );
				xyplot.getDomainAxis().setVisible(i == rindex + mNumRows - 1);
				
				mChartsInnerPanel.add(cp);
			}
			
			if (mShowAverage) {
				ChartPanel cp = mAverages[i];
				JFreeChart jfc = cp.getChart();
				
				if (i == rindex) {
					jfc.setTitle("Average");
				}
				else {
					jfc.setTitle("");
				}
				
				XYPlot xyplot = jfc.getXYPlot();
				xyplot.getRangeAxis().setLabel("");
				xyplot.getDomainAxis().setVisible(i == rindex + mNumRows - 1);
				
				mChartsInnerPanel.add(cp);
			}
		}
	}
	
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == mAttachDetachButton) {
			super.actionPerformed(e);
		} else if (e.getSource() == mFileComboBox) {
			int file = mFileComboBox.getSelectedIndex();
			updateLvComboBoxAndTabs(file);
		
		} else {
			((JSpinner.DefaultEditor) mNumRowsTextField.getEditor()).
					getTextField().setEditable(false);
			((JSpinner.DefaultEditor) mNumColsTextField.getEditor()).
					getTextField().setEditable(false);
			mNumRows = (Integer)mNumRowsTextField.getValue();
			mNumCols = (Integer)mNumColsTextField.getValue();
			mShowAverage = mShowAverageCheckBox.isSelected();
		
			int group = mGroupComboBox.getSelectedIndex();
			int max = mXAxisNames.get(group).size();
			mColsSpinner.setMaximum(max);
		
			if (mNumCols > max) {
				mNumColsTextField.setValue(max);
				mNumCols = max;
			}
		
			rearrangePlots();
			getParent().repaint();
		}
	}

	public void adjustmentValueChanged(AdjustmentEvent event) {
		int rindex = mRowScrollBar.getValue();
		int cindex = mColScrollBar.getValue();
		
		layoutPlots(rindex, cindex);
		getParent().repaint();
	}
	
	public void doSaveAs() {
		// Checks if any charts are currently being displayed first.
		if (mChartsPanel.getComponentCount() == 0) {
			GlobalVariablesFunctions.showErrorMessage(
					"No plots have been loaded yet.");
			return;
		}
		
		int selectedIndex = mGroupComboBox.getSelectedIndex();
		int rowValue = mRowScrollBar.getValue();
		int colValue = mColScrollBar.getValue();
		
		ImageWriter.saveAFPlot(this, 
							selectedIndex, 
							rowValue, 
							colValue, 
							mCharts, 
							mShowAverage,
							mXAxisNames,
							mAverages);
	}
	
}