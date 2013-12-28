package pls.chrome.result.view;

import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;

import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JTextField;
import javax.swing.SpringLayout;
import javax.swing.border.LineBorder;

import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.NumberTickUnit;
import org.jfree.chart.labels.StandardXYToolTipGenerator;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.StandardXYItemRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import pls.chrome.result.controller.ResultsCommandManager;
import pls.chrome.result.controller.observer.Event;
import pls.chrome.result.controller.observer.selection.SelectedDataTypeChangedEvent;
import pls.chrome.result.controller.observer.selection.SelectedLvChangedEvent;
import pls.chrome.result.controller.observer.selection.SelectedVolumeChangedEvent;
import pls.chrome.result.controller.observer.selection.SelectionEvent;
import pls.chrome.result.controller.observer.selection.SelectionObserver;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.PlsResultModel;
import pls.chrome.result.model.ResultModel;
import pls.chrome.shared.FilePathCheck;
import pls.shared.GlobalVariablesFunctions;

import com.jmatio.io.MatFileFilter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;

import extern.NewMatFileReader;

/**
 * This plot shows the intensity that each subject had for each condition
 * for a given voxel.
 */
@SuppressWarnings("serial")
public class VoxelIntensityResponsePlot extends AbstractPlot
implements SelectionObserver {
	
	private String mCurrentFilePath;
	private JButton mGoButton = new JButton("Go");
	
	private JComboBox mGroupComboBox = new JComboBox();
	private ArrayList<ArrayList<String>> mDatamatProfileNames;
	private String mFileDir = null;
	private ArrayList<ArrayList<String>> mSubjectNames;
	
	private JTextField mXfield = new JTextField(3);
	private JTextField mYfield = new JTextField(3);
	private JTextField mZfield = new JTextField(3);
	
	private JPanel mConditionsPanel = null;
	private JPanel mLeftPanel = new JPanel();
	private JPanel mRightPanel = new JPanel();
	
	private JLabel mOutOfDateLabel = new JLabel("This plot is out of date.  Click 'Go' to replot.");
	
	private int mX = 0, mY = 0, mZ = 0;
	
	private JProgressBar mProgressBar = new JProgressBar(JProgressBar.HORIZONTAL);
	
	// A boolean array indicating which groups have had all their
	// data successfully loaded.
	private boolean[] mDataLoaded;
	
	public VoxelIntensityResponsePlot(String title, GeneralRepository repository) {
		super(title, repository, GlobalVariablesFunctions.PLS);
		
		// Register this plot's observer
		repository.getPublisher().registerObserver(this);
		
		// This plot is not using the layout that was setup in AbstractPlot,
		// but its widgets are still being used though.
		removeAll();
		
		setupWidgets();
	}
	
	/**
	 * Sets up the widgets for this plot panel.
	 */
	private void setupWidgets() {
		setLayout(new BoxLayout(this, BoxLayout.X_AXIS));
		
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

		width = mGroupComboBox.getPreferredSize().width + 70;
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
		
		// Set up the controls for entering voxel coordinates.
		JPanel coordinatesPanel = new JPanel();
		coordinatesPanel.setAlignmentX(JPanel.LEFT_ALIGNMENT);
		s = new SpringLayout();
		coordinatesPanel.setLayout(s);
		
		JLabel coordinatesLabel = new JLabel("Voxel Coordinates:");
		coordinatesPanel.add(coordinatesLabel);
		coordinatesPanel.add(mXfield);
		coordinatesPanel.add(mYfield);
		coordinatesPanel.add(mZfield);
		coordinatesPanel.add(mGoButton);
		
		s.putConstraint(SpringLayout.NORTH, coordinatesLabel, 30, SpringLayout.NORTH, coordinatesPanel);
		s.putConstraint(SpringLayout.WEST, coordinatesLabel, 5, SpringLayout.WEST, coordinatesPanel);
		s.putConstraint(SpringLayout.NORTH, mXfield, 0, SpringLayout.SOUTH, coordinatesLabel);
		s.putConstraint(SpringLayout.WEST, mXfield, 5, SpringLayout.WEST, coordinatesPanel);
		s.putConstraint(SpringLayout.NORTH, mYfield, 0, SpringLayout.SOUTH, coordinatesLabel);
		s.putConstraint(SpringLayout.WEST, mYfield, 5, SpringLayout.EAST, mXfield);
		s.putConstraint(SpringLayout.NORTH, mZfield, 0, SpringLayout.SOUTH, coordinatesLabel);
		s.putConstraint(SpringLayout.WEST, mZfield, 5, SpringLayout.EAST, mYfield);
		s.putConstraint(SpringLayout.NORTH, mGoButton, -3, SpringLayout.SOUTH, coordinatesLabel);
		s.putConstraint(SpringLayout.WEST, mGoButton, 5, SpringLayout.EAST, mZfield);
		
		width = mXfield.getPreferredSize().width
				  + mYfield.getPreferredSize().width
				  + mZfield.getPreferredSize().width
				  + mGoButton.getPreferredSize().width
				  + 20;
		height = coordinatesLabel.getPreferredSize().height
				   + mGoButton.getPreferredSize().height
				   + 30;
		dimension = new Dimension(width, height);
		coordinatesPanel.setPreferredSize(dimension);
		coordinatesPanel.setMaximumSize(dimension);
		coordinatesPanel.setMinimumSize(dimension);
		
		// Set up the progress bar.
		JPanel progressPanel = new JPanel();
		progressPanel.setAlignmentX(JPanel.LEFT_ALIGNMENT);
		s = new SpringLayout();
		progressPanel.setLayout(s);
		progressPanel.add(mProgressBar);
		
		s.putConstraint(SpringLayout.NORTH, mProgressBar, 0, SpringLayout.NORTH, progressPanel);
		s.putConstraint(SpringLayout.WEST, mProgressBar, 5, SpringLayout.WEST, progressPanel);

		width = mProgressBar.getPreferredSize().width + 5;
		height = mProgressBar.getPreferredSize().height;
		dimension = new Dimension(width, height);
		progressPanel.setPreferredSize(dimension);
		progressPanel.setMaximumSize(dimension);
		progressPanel.setMinimumSize(dimension);
		
		mGoButton.addActionListener(this);
		mGroupComboBox.addActionListener(this);

		mLeftPanel.setLayout(new BoxLayout(mLeftPanel, BoxLayout.Y_AXIS));
		mLeftPanel.setAlignmentY(JPanel.TOP_ALIGNMENT);
		mLeftPanel.add(buttonPanel);
		mLeftPanel.add(filePanel);
		mLeftPanel.add(groupPanel);
		mLeftPanel.add(coordinatesPanel);
		mLeftPanel.add(progressPanel);
		
		mRightPanel.setAlignmentY(JPanel.TOP_ALIGNMENT);
		
		add(mLeftPanel);
		add(mRightPanel);
	}
	
	protected void updateLvComboBoxAndTabs(int fileIndex) {
		PlsResultModel model = mRepository.getPlsModel(mResultFilePaths.get(fileIndex));

		mSubjectNames = model.getSubjectNames();
		mDatamatProfileNames = model.getDatamatProfiles();
		mFileDir = model.getFileDir();
		
		if (mGroupComboBox == null) {
			mGroupComboBox = new JComboBox();
		} else {
			mGroupComboBox.removeActionListener(this);
			mGroupComboBox.removeAllItems();
		}
		mDataLoaded = new boolean[mDatamatProfileNames.size()];
		for (int i = 0; i < mDatamatProfileNames.size(); i++) {
			mGroupComboBox.addItem("Group " + (i + 1));
			mDataLoaded[i] = true;
		}
		mGroupComboBox.addActionListener(this);
		
		int[] voxel = model.getSelectionModel().getSelectedVoxel();
		if (mXfield == null) {
			mXfield = new JTextField(3);
			mYfield = new JTextField(3);
			mZfield = new JTextField(3);
		}
		mXfield.setText(Integer.toString(voxel[0]));
		mYfield.setText(Integer.toString(voxel[1]));
		mZfield.setText(Integer.toString(voxel[2]));
		
		// Set up the legend for identifying the conditions.
		if (mConditionsPanel != null) {
			mLeftPanel.remove(mConditionsPanel);
		}
		mConditionsPanel = new JPanel();
		mConditionsPanel.setAlignmentX(JPanel.LEFT_ALIGNMENT);
		SpringLayout s = new SpringLayout();
		mConditionsPanel.setLayout(s);
		
		JPanel conditions = new JPanel();
		conditions.setBorder(new LineBorder(Color.DARK_GRAY, 1) );
		conditions.add(new JLabel(" Conditions: "));
		conditions.setLayout(new BoxLayout(conditions, BoxLayout.Y_AXIS));
		
		ArrayList<String> conditionNames = model.getConditionNames();
		for (int i = 0; i < conditionNames.size(); ++i) {
			conditions.add(new JLabel(" (" + (i + 1) + ") " + conditionNames.get(i) ) );
		}
		mConditionsPanel.add(conditions);
		
		s.putConstraint(SpringLayout.NORTH, conditions, 40, SpringLayout.NORTH, mConditionsPanel);
		s.putConstraint(SpringLayout.WEST, conditions, 5, SpringLayout.WEST, mConditionsPanel);
		
		int width = conditions.getPreferredSize().width + 5;
		int height = conditions.getPreferredSize().height + 40;
		Dimension dimension = new Dimension(width, height);
		mConditionsPanel.setPreferredSize(dimension);
		mConditionsPanel.setMaximumSize(dimension);
		mConditionsPanel.setMinimumSize(dimension);
		
		if (mLeftPanel == null) {
			mLeftPanel = new JPanel();
		}
		mLeftPanel.add(mConditionsPanel);
	}
	
	// This function is returning null since the min and max range values
	// are not required here.
	public double[][] getRangeData(ResultModel model) {
		return null;
	}
	
	// This function is not being used here since the charts are not
	// being automatically generated whenever a user switches result files.
	public void makeChart(int fileIndex) {}
	
	/**
	 * Loads the datamat files necessary to replot the voxel intensity
	 * response plot for the requested group and voxel.
	 */
	private void createChart(int group, int x, int y, int z) {
		ArrayList<String> datamatNames = mDatamatProfileNames.get(group);
		
		mProgressBar.setValue(0);
		mProgressBar.setMinimum(0);
		mProgressBar.setMaximum(datamatNames.size() * 2 + 2);
		
		Map<String, MLArray> currFile = null;
		
		XYSeriesCollection dataset = new XYSeriesCollection();
		StandardXYToolTipGenerator tooltip = new StandardXYToolTipGenerator();
		StandardXYItemRenderer renderer = new StandardXYItemRenderer(StandardXYItemRenderer.SHAPES, tooltip);
		
		int numConditions = 0;
		
		// The entire group is made unavailable if all the datamat files in
		// that group are missing.
		if (datamatNames.isEmpty()) {
			GlobalVariablesFunctions.showErrorMessage("No datamat files were found for Group " + (group + 1) + ".");
    		removeGroupAvailability(group);
    		return;
		}
		
		FilePathCheck check = new FilePathCheck();
		for (int i = 0; i < datamatNames.size(); ++i) {
			String datamatName = datamatNames.get(i);
			datamatName = check.getExistingFilePath("datamat", datamatName, mFileDir);
			
			if (datamatName == null) {
				GlobalVariablesFunctions.showErrorMessage("Not all datamat files were found for Group " + (group + 1) + ".");
	    		removeGroupAvailability(group);
	    		return;
	    	}
			
			try {
				currFile = new NewMatFileReader(datamatName, new MatFileFilter()).getContent();
			}
			catch (IOException e) {
				GlobalVariablesFunctions.showErrorMessage("Datamat information could not be read from file " + datamatName + ".");
				removeGroupAvailability(group);
				return;
			} 
			
			double[][] stDatamat = ((MLDouble)currFile.get("st_datamat")).getArray();
			double[] stCoords = ((MLDouble)currFile.get("st_coords")).getArray()[0];
			double[] stDims = ((MLDouble)currFile.get("st_dims")).getArray()[0];
			
			XYSeries series = new XYSeries(mSubjectNames.get(group).get(i));
			
			numConditions = stDatamat.length;
			
			// For each condition
			for (int row = 0; row < stDatamat.length; ++row) {
				// we map into 1-D coordinates
				int index = (int)((stDims[0] * stDims[1] * (z-1)) + (stDims[0] * (y-1)) + (x-1));
				
				int j = 0;
				boolean found = false;
				
				for (j = 0; j < stCoords.length; ++j) {
					if (stCoords[j] == index + 1) {
						found = true;
						break;
					}
				}
				
				double d = 0;
				if (found) {
					d = stDatamat[row][j];
				}
				
				series.add(row + 1, d);
			}
			
			dataset.addSeries(series);
			
			mProgressBar.setValue(mProgressBar.getValue() + 2);
		}
		
		NumberAxis domainAxis = new NumberAxis("Condition");
		domainAxis.setTickUnit(new NumberTickUnit(1.0) );
		domainAxis.setRange(0.1, numConditions + 0.9);
		NumberAxis rangeAxis = new NumberAxis("Intensity");
		
		
		XYPlot plot = new XYPlot(dataset, domainAxis, rangeAxis, renderer);
		
		JFreeChart chart = new JFreeChart("Voxel Intensity Response", plot);
		
		mChartPanel.setChart(chart);
		mProgressBar.setValue(mProgressBar.getMaximum());
	}
	
	private void removeGroupAvailability(int group) {
		mDataLoaded[group] = false;
		
		String text = (String) mGroupComboBox.getItemAt(group);
		text += " (unavailable)";
		mGroupComboBox.insertItemAt(text, group);
		mGroupComboBox.removeItemAt(group + 1);
	}

	public void actionPerformed(ActionEvent e) {
		final int group = mGroupComboBox.getSelectedIndex();
		
		if (!mDataLoaded[group]) {
			return;
		}
		
		if (e.getSource() == mAttachDetachButton) {
			super.actionPerformed(e);
		} else if (e.getSource() == mFileComboBox) {
			mRightPanel.remove(mChartPanel);
			mRightPanel.add(mOutOfDateLabel);
			
			int file = mFileComboBox.getSelectedIndex();
			mCurrentFilePath = mResultFilePaths.get(file);
			ResultsCommandManager.selectResultFile(mCurrentFilePath);
			updateLvComboBoxAndTabs(file);
			
			repaint();
		
		} else {
			mChartPanel.setVisible(false);

			mX = Integer.parseInt(mXfield.getText() );
			mY = Integer.parseInt(mYfield.getText() );
			mZ = Integer.parseInt(mZfield.getText() );
			ResultsCommandManager.selectVoxel(0, mX, mY, mZ);
			
			new Thread() {
				public void run() {
					mGoButton.setEnabled(false);
					createChart(group, mX, mY, mZ);
//					mRightPanel.add(mChartPanel);
		
					getParent().repaint();
					
					mChartPanel.setVisible(true);
					mGoButton.setEnabled(true);
				}
			}.start();
			
		}
	}
	
	public void updateSelection() {
		String filePath = mRepository.getSelectedResultFile();
		String fileName = filePath.substring(filePath.lastIndexOf(File.separator) + 1, filePath.length());
		mFileComboBox.setSelectedItem(fileName);
		
		int[] selectedVoxel = mRepository.getGeneral().getSelectionModel().getSelectedVoxel();
		int x = selectedVoxel[0];
		int y = selectedVoxel[1];
		int z = selectedVoxel[2];
		
		mXfield.setText(Integer.toString(x));
		mYfield.setText(Integer.toString(y));
		mZfield.setText(Integer.toString(z));
		
		if (mX != x || mY != y || mZ != z) {
			mRightPanel.remove(mChartPanel);
			mRightPanel.add(mOutOfDateLabel);
		}
		else {
			mRightPanel.remove(mOutOfDateLabel);
			mRightPanel.add(mChartPanel);
		}
	}
	
	public void doSaveAs() {
		if (mRightPanel.getComponentCount() != 0) {
			Component component = mRightPanel.getComponent(0);
			if (component == mChartPanel) {
				doSaveAs(mChartPanel);
				return;
			}
		}
		
		GlobalVariablesFunctions.showErrorMessage("No plot has been loaded yet.");
	}
	
	protected boolean ModelIsApplicable(ResultModel model) {
		
		// This plot is only wanted for blocked fMRI result files,
		// i.e. there is only one lag.
		return (model.getWindowSize() == 1);
	}

	@Override
	public void notify(SelectedDataTypeChangedEvent e) {}

	@Override
	public void notify(SelectedLvChangedEvent e) {}

	@Override
	public void notify(SelectedVolumeChangedEvent e) {}

	@Override
	public void notify(SelectionEvent e) {
		updateSelection();
	}

	@Override
	public void notify(Event e) {}
}
