package pls.chrome.result.view;

import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;

import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JTextField;
import javax.swing.SpringLayout;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.NumberTickUnit;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import pls.chrome.result.controller.ResultsCommandManager;
import pls.chrome.result.controller.observer.Event;
import pls.chrome.result.controller.observer.Observer;
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

@SuppressWarnings("serial")
public class ResponseFunctionPlot extends AbstractFunctionalPlot
implements SelectionObserver {
	
	private ArrayList<ArrayList<String>> mDatamatProfileNames;
	private String mFileDir;
	private JTextField mXfield = new JTextField(3);
	private JTextField mYfield = new JTextField(3);
	private JTextField mZfield = new JTextField(3);
	
	private JButton mGoButton = new JButton("Go");
	private int mX = 0;
	private int mY = 0;
	private int mZ = 0;
	private int mLag = 0;
	
	private String mCurrentFile;
	
	// A boolean array indicating which groups have had all their
	// data successfully loaded.
	private boolean[] mDataLoaded;
	
	private JLabel mOutOfDateLabel = new JLabel("This plot is out of date.  Click 'Go' to replot.");
	
	public ResponseFunctionPlot(String title, GeneralRepository repository) {
		super(title, repository, GlobalVariablesFunctions.PLS, false);
		
		repository.getPublisher().registerObserver(this);
		
		initializeWidgets();
		
		// Sets the first file in the newly-initialized file combo box as
		// the currently selected file.
		mCurrentFile = (String) mFileComboBox.getItemAt(0);
	}
	
	private void initializeWidgets() {
		JPanel coordinatesPanel = new JPanel();
		coordinatesPanel.setAlignmentX(JPanel.LEFT_ALIGNMENT);
		SpringLayout s = new SpringLayout();
		coordinatesPanel.setLayout(s);
		
		JLabel coordinatesLabel = new JLabel("Voxel Coordinates:");
		coordinatesPanel.add(coordinatesLabel);
		coordinatesPanel.add(mXfield);
		coordinatesPanel.add(mYfield);
		coordinatesPanel.add(mZfield);
		coordinatesPanel.add(mGoButton);
		
		s.putConstraint(SpringLayout.NORTH, coordinatesLabel, 40, SpringLayout.NORTH, coordinatesPanel);
		s.putConstraint(SpringLayout.WEST, coordinatesLabel, 5, SpringLayout.WEST, coordinatesPanel);
		s.putConstraint(SpringLayout.NORTH, mXfield, 0, SpringLayout.SOUTH, coordinatesLabel);
		s.putConstraint(SpringLayout.WEST, mXfield, 5, SpringLayout.WEST, coordinatesPanel);
		s.putConstraint(SpringLayout.NORTH, mYfield, 0, SpringLayout.SOUTH, coordinatesLabel);
		s.putConstraint(SpringLayout.WEST, mYfield, 5, SpringLayout.EAST, mXfield);
		s.putConstraint(SpringLayout.NORTH, mZfield, 0, SpringLayout.SOUTH, coordinatesLabel);
		s.putConstraint(SpringLayout.WEST, mZfield, 5, SpringLayout.EAST, mYfield);
		s.putConstraint(SpringLayout.NORTH, mGoButton, -3, SpringLayout.SOUTH, coordinatesLabel);
		s.putConstraint(SpringLayout.WEST, mGoButton, 5, SpringLayout.EAST, mZfield);
		
		int width = mXfield.getPreferredSize().width
				  + mYfield.getPreferredSize().width
				  + mZfield.getPreferredSize().width
				  + mGoButton.getPreferredSize().width
				  + 20;
		int height = coordinatesLabel.getPreferredSize().height
				   + mGoButton.getPreferredSize().height
				   + 40;
		Dimension dimension = new Dimension(width, height);
		coordinatesPanel.setPreferredSize(dimension);
		coordinatesPanel.setMaximumSize(dimension);
		coordinatesPanel.setMinimumSize(dimension);
		
		mGoButton.addActionListener(this);
		
		mLeftPanel.add(coordinatesPanel);
		mLeftPanel.add(mProgressPanel);
	}
	
	protected void updateLvComboBoxAndTabs(int fileIndex) {
		super.updateLvComboBoxAndTabs(fileIndex);
		PlsResultModel model = mRepository.getPlsModel(mResultFilePaths.get(fileIndex));
		
		mDatamatProfileNames = model.getDatamatProfiles();
		mFileDir = model.getFileDir();
		
		mDataLoaded = new boolean[mDatamatProfileNames.size()];
		for (int i = 0; i < mDatamatProfileNames.size(); ++i) {
			mDataLoaded[i] = true;
		}
		
		int[] voxel = model.getSelectionModel().getSelectedVoxel();
		if (mXfield == null) {
			mXfield = new JTextField(3);
			mYfield = new JTextField(3);
			mZfield = new JTextField(3);
		}
		mXfield.setText(Integer.toString(voxel[0]));
		mYfield.setText(Integer.toString(voxel[1]));
		mZfield.setText(Integer.toString(voxel[2]));
	}

	protected void createCharts() {
		int group = mGroupComboBox.getSelectedIndex();
		
		ArrayList<String> datamatNames = mDatamatProfileNames.get(group);
		
		mCharts = new ChartPanel[mYAxisNames.size()][datamatNames.size()];
		mAverages = new ChartPanel[mYAxisNames.size()];
		
		if (mProgressBar == null) {
			mProgressBar = new JProgressBar(JProgressBar.HORIZONTAL);
		}
		mProgressBar.setValue(0);
		mProgressBar.setMinimum(0);
		mProgressBar.setMaximum(datamatNames.size() * 2 + 2);
		
		// First we got to load the session profile
		Map<String, MLArray> currFile = null;
		
		int numConditions = 0;
		int stWinSize = 0;
		
		// The entire group is made unavailable if all the datamat files in
		// that group are missing.
		if (datamatNames.isEmpty()) {
    		JOptionPane.showMessageDialog(null, "No datamat files were found for Group " + (group + 1) + ".", "Error", JOptionPane.ERROR_MESSAGE);
    		removeGroupAvailability(group);
    		return;
		}
		
		ArrayList<double[]> averages = new ArrayList<double[]>();
		FilePathCheck check = new FilePathCheck();
		for (int i = 0; i < datamatNames.size(); i++) {
			String datamatName = datamatNames.get(i);
			datamatName = check.getExistingFilePath("datamat", datamatName, mFileDir);
			
			if (datamatName == null) {
	    		JOptionPane.showMessageDialog(null, "Not all datamat files were found for Group " + (group + 1) + ".", "Error", JOptionPane.ERROR_MESSAGE);
	    		removeGroupAvailability(group);
	    		return;
	    	}
			
			// Read in the datamat file
			try {
				currFile = new NewMatFileReader(datamatName, new MatFileFilter()).getContent();
			}
			catch (IOException e) {
				JOptionPane.showMessageDialog(null, "Datamat information could not be read from file " + datamatName + ".", "Error", JOptionPane.ERROR_MESSAGE);
				removeGroupAvailability(group);
				return;
			}
			
			// Obtain the variables we need from the datamat file
			double[][] stDatamat = ((MLDouble)currFile.get("st_datamat")).getArray();
			double[] stCoords = ((MLDouble)currFile.get("st_coords")).getArray()[0];
			double[] stDims = ((MLDouble)currFile.get("st_dims")).getArray()[0];
			stWinSize = ((MLDouble)currFile.get("st_win_size")).getReal(0, 0).intValue();
			
			numConditions = mYAxisNames.size();
			
			// For each condition
			for (int row = 0; row < numConditions; ++row) {
				if (averages.size() <= row) {
					averages.add(new double[stWinSize]);
				}
				
				XYSeriesCollection dataset = new XYSeriesCollection();
				XYSeries series = new XYSeries(mXAxisNames.get(group).get(i) + "/" + mYAxisNames.get(row) );

				// we map into 1-D coordinates
				int index = (int)((stDims[0] * stDims[1] * (mZ-1)) + (stDims[0] * (mY-1)) + (mX-1));
				
				// see if the selected voxel is a "significant" voxel
				// (i.e. see if it is in st_coords)
				int j = 0;
				boolean found = false;
				
				for (j = 0; j < stCoords.length; ++j) {
					if (stCoords[j] == index + 1) {
						found = true;
						break;
					}
				}
				
				// If we find it, plot the value for each lag
				if (found) {
					int temporalIndex = j * stWinSize;
					
					
					// For each lag
					for (int lag = 0; lag < stWinSize; ++lag) {
						double d = stDatamat[row][temporalIndex + lag];
						series.add(lag, d);
						averages.get(row)[lag] += d;
					}
				}
				else {
					for (int lag = 0; lag < stWinSize; ++lag) {
						series.add(lag, 0.0);
					}
				}
				
				dataset.addSeries(series);
				
				JFreeChart chart = ChartFactory.createXYLineChart(mXAxisNames.get(group).get(i) + "/" + mYAxisNames.get(row),
						"Lag", "Intensity", dataset, PlotOrientation.VERTICAL, false, true, false);
				
				NumberAxis domainAxis = new NumberAxis("Lag");
				domainAxis.setTickUnit(new NumberTickUnit(1.0) );
				chart.getXYPlot().setDomainAxis(domainAxis);
				
				mCharts[row][i] = new ChartPanel(chart);
			}
			
			mProgressBar.setValue(mProgressBar.getValue() + 2);
		}
		
		for (int cond = 0; cond < averages.size(); ++cond) {
			XYSeriesCollection dataset = new XYSeriesCollection();
			XYSeries series = new XYSeries("Average");
			for (int j = 0; j < averages.get(cond).length; ++j) {
				series.add(j, averages.get(cond)[j] / stWinSize);
			}
			dataset.addSeries(series);
			
			JFreeChart chart = ChartFactory.createXYLineChart("Average", "Lag", "Intensity", dataset,
					PlotOrientation.VERTICAL, false, true, false);
			
			NumberAxis domainAxis = new NumberAxis("Lag");
			domainAxis.setTickUnit(new NumberTickUnit(1.0) );
			chart.getXYPlot().setDomainAxis(domainAxis);
			
			mAverages[cond] = new ChartPanel(chart);
		}
	}

	protected void adjustDomainAndRange() {
		int group = mGroupComboBox.getSelectedIndex();
		if (!mDataLoaded[group]) {
			return;
		}
		
		super.adjustDomainAndRange();
	}
	
	private void removeGroupAvailability(int group) {
		mDataLoaded[group] = false;
		
		String text = (String) mGroupComboBox.getItemAt(group);
		text += " (unavailable)";
		mGroupComboBox.insertItemAt(text, group);
		mGroupComboBox.removeItemAt(group + 1);
	}
	
	public void actionPerformed(final ActionEvent event) {
		int group = mGroupComboBox.getSelectedIndex();
		if (!mDataLoaded[group]) {
			return;
		}
		
		if (event.getSource() == mFileComboBox) {
			mChartsPanel.removeAll();
			mChartsPanel.add(mOutOfDateLabel, BorderLayout.CENTER);
			
			int file = mFileComboBox.getSelectedIndex();
			ResultsCommandManager.selectResultFile(mResultFilePaths.get(file) );
			repaint();
			super.actionPerformed(event);
		} else if (event.getSource() == mGoButton || event.getSource() == mGroupComboBox) {

			mChartsPanel.setVisible(false);
			
			mX = Integer.parseInt(mXfield.getText() );
			mY = Integer.parseInt(mYfield.getText() );
			mZ = Integer.parseInt(mZfield.getText() );
			ResultsCommandManager.selectVoxel(mLag, mX, mY, mZ);
			
			
			// Creates the charts on a separate thread such that the progess
			// bar used by it can be displayed properly.
			new Thread() {
				public void run() {
					mGoButton.setEnabled(false);
					
					initializeCharts();
					superActionPerformed(event);

					mGoButton.setEnabled(true);
					mChartsPanel.setVisible(true);
				}
			}.start();
		} else {
			super.actionPerformed(event);
		}
	}
	
	// This method was written such that it can be called from
	// inside a new thread.
	private void superActionPerformed(ActionEvent e) {
		super.actionPerformed(e);
	}
	
	public void updateSelection() {
		String filePath = mRepository.getSelectedResultFile();
		String fileName = filePath.substring(filePath.lastIndexOf(File.separator) + 1, filePath.length());
		
		if (!fileName.equals(mCurrentFile)) {
			mFileComboBox.setSelectedItem(fileName);
			mCurrentFile = fileName;
		}
		
		PlsResultModel model = mRepository.getPlsModel(mCurrentFile);
		
		if (model != null) {
			int[] selectedVoxel = model.getSelectionModel().getSelectedVoxel();
			int x = selectedVoxel[0];
			int y = selectedVoxel[1];
			int z = selectedVoxel[2];
			mLag = selectedVoxel[3];
			
			mXfield.setText(Integer.toString(x));
			mYfield.setText(Integer.toString(y));
			mZfield.setText(Integer.toString(z));
			
			mChartsPanel.removeAll();
			if (mX != x || mY != y || mZ != z) {
				mChartsPanel.add(mOutOfDateLabel, BorderLayout.CENTER);
			}
			else {
				rearrangePlots();
			}
		}
	}
	
	protected void rearrangePlots() {
		int group = mGroupComboBox.getSelectedIndex();
		if (!mDataLoaded[group]) {
			return;
		}
		
		super.rearrangePlots();
	}
	
	public void doSaveAs() {
		if (mChartsPanel.getComponentCount() != 0) {
			Component component = mChartsPanel.getComponent(0);
			if (component != mOutOfDateLabel) {
				super.doSaveAs();
				return;
			}
		}
		
		GlobalVariablesFunctions.showErrorMessage("No plots have been loaded yet.");
	}
	
	protected boolean ModelIsApplicable(ResultModel model) {
		boolean sessionEmpty = false;

		if(model.getSessionProfiles() == null)
			sessionEmpty = true;
		
		return (model.getWindowSize() > 1) && !sessionEmpty;
	}
	
	///////////////////////////////////////////////////////////////////////////
	// Event handling
	///////////////////////////////////////////////////////////////////////////
	@Override
	public void notify(Event e) {}

	///////////////////////////////////////////////////////////////////////////
	// Selection event handlers
	///////////////////////////////////////////////////////////////////////////
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
}