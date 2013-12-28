package pls.chrome.result.clusterreport;

import java.awt.BorderLayout;
import java.awt.Font;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashSet;
import java.util.Vector;

import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.JTextField;
import javax.swing.ListSelectionModel;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;
import javax.swing.filechooser.FileFilter;

import pls.chrome.result.controller.ResultsCommandManager;
import pls.chrome.result.controller.observer.Event;
import pls.chrome.result.controller.observer.Observer;
import pls.chrome.result.controller.observer.colourscale.ColourScaleEvent;
import pls.chrome.result.controller.observer.colourscale.ColourScaleObserver;
import pls.chrome.result.controller.observer.selection.SelectedDataTypeChangedEvent;
import pls.chrome.result.model.BrainData;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.ResultModel;
import pls.shared.GlobalVariablesFunctions;

/**
 * This panel allows the creation and display, and saving of a cluster report.
 * It contains options to customize the report, such as minimum cluster size
 * and the minimum distance between cluster peaks.  It is from here that a
 * cluster mask may be applied, and where the cluster report may be unloaded.
 */
@SuppressWarnings("serial")
public class ClusterReportPanel extends JPanel implements ActionListener,
ListSelectionListener, ColourScaleObserver {
	// These variables are used to determine if the report is out of sync
	// with the currently applied threshold (generating a report takes too
	// long to keep the report in sync in real time).
	double mThreshold = 0.0;
	double mUsedThreshold = 0.0;
	
	String mWrongThreshold = "Current threshold is not the same as threshold used to generate report.  Please regenerate the report.";
	
	// Various buttons
	JButton mGenerateButton = null;
	JButton mSaveButton = null;
	JButton mUnloadReportButton = null;
	
	// Various input fields
	JTextField mClusterSizeTextField = null;
	JTextField mClusterDistanceTextField = null;
	JTextField mOriginLocationXTextField = null;
	JTextField mOriginLocationYTextField = null;
	JTextField mOriginLocationZTextField = null;
	JTextField mFilenameTextField = null;
	
	JButton mFilenameBrowseButton = null;
	JPanel mReportPanel = null;
	JLabel mNoReportLoadedLabel = null;
	JTable mReportTable = null;
	JLabel mReportOutOfDateLabel = null;
	
	GeneralRepository mRepository = null;
	
	Vector<HashSet<Cluster> > mClusters = null;
	
	public ClusterReportPanel(GeneralRepository repository) {
		mRepository = repository;
		
		repository.getPublisher().registerObserver(this);
		
		BrainData bData = repository.getGeneral().getBrainData();
		mThreshold = bData.getMaxMinThresh()[2];
		
		setLayout(new BoxLayout(this, BoxLayout.PAGE_AXIS) );
		
		JPanel optionsPanel = new JPanel();

		// Create the buttons used to do stuff
		mGenerateButton = new JButton("Create Report");
		mSaveButton = new JButton("Save Report");
		mUnloadReportButton = new JButton("Unload Report");
		
		// Set this object as a listener
		mGenerateButton.addActionListener(this);
		mSaveButton.addActionListener(this);
		mUnloadReportButton.addActionListener(this);
		
		// Create a panel for the generate new report options
		JPanel generatePanel = new JPanel();
		generatePanel.setLayout(new BoxLayout(generatePanel, BoxLayout.PAGE_AXIS) );
		JLabel generateTitle = new JLabel("Create a new Cluster Report");
		generateTitle.setFont(new Font("Arial", Font.ITALIC, 14) );
		generatePanel.add(generateTitle);
		generatePanel.add(new JLabel("Minimum cluster size (in voxels)") );
		mClusterSizeTextField = new JTextField("5");
		generatePanel.add(mClusterSizeTextField);
		generatePanel.add(new JLabel("Minimum distance (in mm) between cluster peaks") );
		mClusterDistanceTextField = new JTextField("10");
		generatePanel.add(mClusterDistanceTextField);
		generatePanel.add(new JLabel("Origin location (in voxels)") );
		generatePanel.add(mGenerateButton);
		
		// Create a panel for the load existing report options
		JPanel savePanel = new JPanel();
		savePanel.setLayout(new BoxLayout(savePanel, BoxLayout.PAGE_AXIS) );
		JLabel saveTitle = new JLabel("Save the current Cluster Report");
		saveTitle.setFont(new Font("Arial", Font.ITALIC, 14) );
		savePanel.add(saveTitle);
		mFilenameTextField = new JTextField();
		mFilenameBrowseButton = new JButton("Browse");
		
		savePanel.add(mSaveButton);
		mSaveButton.setEnabled(false);
		
		// Create a panel for the unload report options
		JPanel unloadPanel = new JPanel();
		unloadPanel.setLayout(new BoxLayout(unloadPanel, BoxLayout.PAGE_AXIS) );
		JLabel unloadTitle = new JLabel("Unload current report");
		unloadTitle.setFont(new Font("Arial", Font.ITALIC, 14) );
		unloadPanel.add(unloadTitle);
		unloadPanel.add(mUnloadReportButton);
		
		// Add all panels to the options panel
		optionsPanel.add(generatePanel);
		optionsPanel.add(savePanel);
		optionsPanel.add(unloadPanel);
		
		// Add the options panel to our main panel
		add(optionsPanel);
		
		// Create the report panel
		mReportPanel = new JPanel();
		mReportPanel.setLayout(new BorderLayout() );
		mNoReportLoadedLabel = new JLabel("No report loaded.");
		mNoReportLoadedLabel.setHorizontalAlignment(JLabel.CENTER);
		mReportOutOfDateLabel = new JLabel("Current threshold is not the same as threshold used to generate report.  Please regenerate the report.");
		mReportOutOfDateLabel.setHorizontalAlignment(JLabel.CENTER);
		mReportPanel.add(mNoReportLoadedLabel);
		
		add(mReportPanel);
	}

	public void actionPerformed(ActionEvent evt) {
		if (evt.getSource() == mGenerateButton)
		{
			mReportPanel.removeAll();
			
			int minimumClusterSize = 5;
			
			try {
				minimumClusterSize = Integer.parseInt(mClusterSizeTextField.getText() );
			}
			catch (NumberFormatException e) {
				GlobalVariablesFunctions.showErrorMessage("The cluster size must be a positive integer.");
				return;
			}
			
			if (minimumClusterSize < 1) {
				GlobalVariablesFunctions.showErrorMessage("The cluster size must be a positive integer.");
				return;
			}
			
			ResultModel model = mRepository.getGeneral();
			
			ClusterCalculator clusterCalc = new ClusterCalculator(model);
			mClusters = clusterCalc.getClusters();
			ClusterReportMaker reportMaker = new ClusterReportMaker(model, mClusters, minimumClusterSize);
			mReportTable = reportMaker.generateReport();
			mReportTable.getSelectionModel().addListSelectionListener(this);
			mReportTable.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
			JScrollPane scroller = new JScrollPane(mReportTable);
			mReportPanel.add(scroller);
			mUsedThreshold = mThreshold;
			mSaveButton.setEnabled(true);
		}
		else if (evt.getSource() == mSaveButton)
		{
			JFileChooser jfc = new JFileChooser();
			jfc.setFileFilter(new FileFilter() {
				public boolean accept(File f) {
			        return f.isDirectory() || f.getName().toLowerCase().endsWith(".csv");
			    }
			
				public String getDescription() {
			        return "Comma Separated Values";
			    }
			});
			int result = jfc.showSaveDialog(this);
			
			if (result == JFileChooser.APPROVE_OPTION) {
				File file = jfc.getSelectedFile();
				String path = file.getAbsolutePath();
				if (!path.endsWith(".csv") ) {
					path += ".csv";
					
					file = new File(path);
				}
				
				try {
					BufferedWriter bw = new BufferedWriter(new FileWriter(file) );
					
					int numRows = mReportTable.getRowCount();
					int numCols = mReportTable.getColumnCount();
					
					for (int i = 0; i < numCols; ++i) {
						String s = mReportTable.getColumnName(i);
						bw.write('\"');
						bw.write(s);
						bw.write('\"');
						if (i < numCols - 1) {
							bw.write(',');
						}
					}
					
					bw.newLine();
					bw.flush();
					
					for (int i = 0; i < numRows; ++i) {
						for (int j = 0; j < numCols; ++j) {
							Object o = mReportTable.getValueAt(i, j);
							String s = o.toString();
							bw.write('\"');
							bw.write(s);
							bw.write('\"');
							if (j < numCols - 1) {
								bw.write(',');
							}
						}
						bw.newLine();
						bw.flush();
					}
					
					bw.close();
				}
				catch (IOException e) {
					System.err.println("An error occured while saving the cluster report");
				}
				
			}
		}
		else if (evt.getSource() == mUnloadReportButton)
		{
			mReportPanel.removeAll();
			mReportPanel.add(mNoReportLoadedLabel);
			
			if (mReportTable != null) {
				mReportTable.getSelectionModel().removeListSelectionListener(this);
			}
			mReportTable = null;
			mClusters = null;
			mUsedThreshold = 0;
			
			mSaveButton.setEnabled(false);
		}
		
		getParent().repaint();
	}

	/**
	 * When the user selects a row (i.e. a cluster), select the peak voxel
	 * of that cluster in all views.
	 */
	public void valueChanged(ListSelectionEvent evt) {
		
		if (!evt.getValueIsAdjusting() ) {
			int lagColumn = 1;
			int coordinateColumn = 2;
			int selectedRow = mReportTable.getSelectedRow();
			
			int lag = (Integer)mReportTable.getValueAt(selectedRow, lagColumn);

			String s = (String)mReportTable.getValueAt(selectedRow, coordinateColumn);
			int pos = s.indexOf(',');
			int x = Integer.parseInt(s.substring(0, pos) );
			int pos2 = s.indexOf(',', pos + 2);
			int y = Integer.parseInt(s.substring(pos + 2, pos2) );
			int z = Integer.parseInt(s.substring(pos2 + 2) );
			
			ResultsCommandManager.selectVoxel(lag, x, y, z);
		}
	}
	
	/**
	 * Set the threshold value, filtering which voxels are included when
	 * calculating clusters.
	 */
	public void updateThreshold() {
		BrainData bData = mRepository.getGeneral().getBrainData();
		mThreshold = bData.getColourScaleModel().getColourScale()[2];
		
		if (mReportTable != null) {
			if (mUsedThreshold != mThreshold) {
				mReportPanel.removeAll();
				mReportPanel.add(mReportOutOfDateLabel);
			}
			else {
				mReportPanel.removeAll();
				JScrollPane scroller = new JScrollPane(mReportTable);
				mReportPanel.add(scroller);
			}
		}
		
		repaint();
	}
	
	///////////////////////////////////////////////////////////////////////////
	// Event handling
	///////////////////////////////////////////////////////////////////////////
	@Override
	public void notify(Event e) {}

	@Override
	public void notify(ColourScaleEvent e) {
		//updateThreshold();
	}

	
}
