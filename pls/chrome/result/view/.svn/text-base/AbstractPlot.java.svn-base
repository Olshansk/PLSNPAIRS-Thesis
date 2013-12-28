package pls.chrome.result.view;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Set;

import javax.imageio.ImageIO;
import javax.swing.ButtonGroup;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.filechooser.FileNameExtensionFilter;

import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.Plot;
import org.jfree.chart.plot.XYPlot;

import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.ResultModel;
import pls.chrome.shared.ImageWriter;
import pls.shared.GlobalVariablesFunctions;

@SuppressWarnings("serial")
public abstract class AbstractPlot extends AttachDetachOption {
	
	protected static final int GLOBAL = 0;
	protected static final int LOCAL = 1;
	
	protected static final int RANGE = 0;
	protected static final int DOMAIN = 1;
        
	//range and domain data. the index is related to the selected index of the
	//jcombobox in the given plot. i.e mRangeData.get(selectedFile)
	protected ArrayList<double[][]> mRangeData;
	protected ArrayList<double[][]> mDomainData;

	protected JPanel mButtonPanel = new JPanel();
	protected JPanel mFilePanel = new JPanel();
	protected JComboBox mFileComboBox = new JComboBox();
	
	public JRadioButton mNoneButton = new JRadioButton("Individual Plot Ranges", true);
	public JRadioButton mLocalButton = new JRadioButton("One Plot Range Per Result File");
	public JRadioButton mGlobalButton = new JRadioButton("One Plot Range For All Result Files");
	
	public ChartPanel mChartPanel = new ChartPanel(null, false);
	
	protected JPanel mSubPane = new JPanel();
	
	protected GeneralRepository mRepository = null;
	protected String mFileType = null;
	
	public double globalRangeMin;
	public double globalRangeMax;
	
	protected double localRangeMin;
	protected double localRangeMax;
	
	protected double globalDomainMin;
	protected double globalDomainMax;
	
	protected double localDomainMin;
	protected double localDomainMax;
	
	protected int itemIndex = 0;
	
	JPanel attachDetachPanel = new JPanel();
	
	public AbstractPlot(String title, GeneralRepository repository, String fileType)
	{
		super(repository, title);
		
		mRepository = repository;
		mFileType = fileType;
		
		// Register the observer for this class.
//		repository.getPublisher().registerObserver(mObserver);

		mButtonPanel.add(mNoneButton);
		mButtonPanel.add(mLocalButton);
		mButtonPanel.add(mGlobalButton);
		
		// Adds the three buttons to a button group so only one of them
		// can be selected at a time.
		ButtonGroup buttonGroup = new ButtonGroup();
		buttonGroup.add(mNoneButton);
		buttonGroup.add(mLocalButton);
		buttonGroup.add(mGlobalButton);
		
		mFilePanel.add(mFileComboBox);
		
		mSubPane.setLayout(new GridLayout(2, 1, 0, 0));
		mSubPane.add(mButtonPanel);
		mSubPane.add(mFilePanel);
		
		JPanel attachDetachPanel = new JPanel();
		attachDetachPanel.add(mAttachDetachButton);
		
		setLayout(new BorderLayout());
		add(mSubPane, BorderLayout.NORTH);
		add(mChartPanel, BorderLayout.CENTER);
		add(attachDetachPanel, BorderLayout.SOUTH);
		
		mNoneButton.addActionListener(this);
		mLocalButton.addActionListener(this);
		mGlobalButton.addActionListener(this);
		
		initialize();
	}
	
	@Override
	public void initialize() {
		
		String selectedItem = null;
		if (mFileComboBox.getItemCount() > 0) {
			selectedItem = (String)mFileComboBox.getSelectedItem();
        }
		
		itemIndex = 0;

		mFileComboBox.removeActionListener(this);
		mFileComboBox.removeAllItems();
		
		calculateGlobalRange();
		
		Set<String> resultFiles = getResultFiles();
		mResultFilePaths = new ArrayList<String>();

		for(String modelName : resultFiles) {
			ResultModel model = mRepository.getGeneral(modelName);
			if (ModelIsApplicable(model) ) {
				mResultFilePaths.add(modelName);
				String fileName = modelName.substring(
						modelName.lastIndexOf(File.separator) + 1, 
						modelName.length());
				
				mFileComboBox.addItem(fileName);
			}
		}
		
		mFileComboBox.setPreferredSize(new Dimension(250, mFileComboBox.getPreferredSize().height));
		itemIndex = lookupFileIndex(selectedItem);

		// Uses the first result file as the default selected result file if the
		//previous selected file was not found.

		if (mFileComboBox.getItemCount() > 0) {
            mFileComboBox.setSelectedIndex(itemIndex);
			particularInit();
			updateLvComboBoxAndTabs(itemIndex); 
			makeChart(itemIndex);
			updateScrollBars(itemIndex);
		}
		mFileComboBox.addActionListener(this);
	}

	/** This is used by charts to set up particular settings that must be
	 *  defined when they are first created and which cannot be done in
	 *  AbstractPlot.java
	 */
	protected void particularInit(){}

	/**
	 * Takes a filename (relative path) and finds its index in the mFileComboBox
	 * @param fileName The filename for which w ewant to find its index.
	 * @return 0 as a default, the index otherwise.
	 */
	protected int lookupFileIndex(String fileName){
		for(int i = 0; i< mFileComboBox.getItemCount(); i++){
			if(mFileComboBox.getItemAt(i).equals(fileName))
				return i;
		}
		return 0;
	}

	/**
	 * Called when a new file is added or an LV is inverted. Calculates the
	 * max and min ranges over all lvs over all files.
	 */
	public void calculateGlobalRange(){
		//get max values
		globalRangeMin = Double.POSITIVE_INFINITY;
		globalRangeMax = Double.NEGATIVE_INFINITY;

		globalDomainMin = Double.POSITIVE_INFINITY;
		globalDomainMax = Double.NEGATIVE_INFINITY;

		mRangeData = new ArrayList<double[][]>();
		mDomainData = new ArrayList<double[][]>();

		Set<String> resultFiles = getResultFiles();
		
		for(String modelName : resultFiles) {
			ResultModel model = mRepository.getGeneral(modelName);

			if (ModelIsApplicable(model) ) {
				//the min/max range/dom over ALL open files is set here.
				//hence we are setting the global range/dom here.
				double[][] rangeData = getRangeData(model);
				mRangeData.add(rangeData);
				getMinMaxValues(rangeData, GLOBAL, RANGE);

				double[][] domainData = getDomainData(model);
				mDomainData.add(domainData);
				getMinMaxValues(domainData, GLOBAL, DOMAIN);
			}
		}
	}

	/**
	 * Retrieves the set of models that have been loaded into the repository.
	 * Retrieves pls modles if we are dealing with a pls file, npairs otherwise.
	 * @return a set of filenames (absolute path) of the models currently loaded.
	 */
	protected Set<String> getResultFiles(){
		Set<String> resultFiles;
		if (mFileType.equals(GlobalVariablesFunctions.PLS)) {
			resultFiles = mRepository.getPlsModels();
		} else {
			resultFiles = mRepository.getNPairsModels();
		}
		return resultFiles;
	}

	protected void updateLvComboBoxAndTabs(int fileIndex) {
		
		// Does nothing here, but can be overriden by plots that
		// have extra combo boxes for CV/LV selection and tabs
		// for group selection.
	}
	
	protected void updateScrollBars(int fileIndex) {
		
		// Does nothing here, but can be overriden by plots that
		// have scrollbars that need to be adjusted.
	}
	
	public abstract void makeChart(int fileIndex);
	
	// This method returns true if there is to be extra white space
	// placed underneath the overall min value of the plot.
	// The default implementation here is applied for bar charts though,
	// and extra white space is only placed underneath if there are
	// negative values in the data.
	// This implementation should be overriden in the case of a
	// box-and-whisker plot or an XY line chart where extra white space
	// should always be placed underneath.
	public boolean addWhiteSpaceAtBottom() {
		if (mGlobalButton.isSelected()) {
			return (globalRangeMin < 0);
		} else if (mLocalButton.isSelected()){
			return (localRangeMin < 0);
		} else {
			return false;
		}
	}
	
	public abstract double[][] getRangeData(ResultModel model);
	
	// This function was not declared abstract since most plots extending this
	// class do not need its max and min domain values to be manually set, and
	// we did not want to implement this function for most plots since they
	// would also be returning null.
	
	public double[][] getDomainData(ResultModel model) {
		return null;
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == mAttachDetachButton) {
			super.actionPerformed(e);
		} else if (e.getSource() == mFileComboBox) {
			int file = mFileComboBox.getSelectedIndex();
			updateLvComboBoxAndTabs(file); //lvcombobox,tabs,charts destroyed.
			makeChart(file);
			updateScrollBars(file);
		} else {
			int file = mFileComboBox.getSelectedIndex();
			makeChart(file);
		}
	}

	/**Finds the min and max values over all CVs/LVs for the range and
	 *
	 * @param data This is either domain or range data for every possible cv.
	 * @param type Indicates whether we want to set the min/max values on a
	 *             global scale (over all result files) or set it on a local
	 *             scale (the currently selected result file)
	 * @param axis Indicates the axis the data refers to. Range or domain.
	 */

	protected void getMinMaxValues(double[][] data, int type, int axis) {
		if (data == null) {
			return;
		}
		
		if (type == GLOBAL && axis == RANGE) {
			for (int i = 0; i != data.length; i++) {
				for (int j = 0; j != data[0].length; j++) {
					double value = data[i][j];
					globalRangeMin = Math.min(value, globalRangeMin);
					globalRangeMax = Math.max(value, globalRangeMax);
				}
			}
		} else if (type == GLOBAL) {
			for (int i = 0; i != data.length; i++) {
				for (int j = 0; j != data[0].length; j++) {
					double value = data[i][j];
					globalDomainMin = Math.min(value, globalDomainMin);
					globalDomainMax = Math.max(value, globalDomainMax);
				}
			}
		} else if (type == LOCAL && axis == RANGE) {
			for (int i = 0; i != data.length; i++) {
				for (int j = 0; j != data[0].length; j++) { //for each cv/lv...
					double value = data[i][j];
					localRangeMin = Math.min(value, localRangeMin);
					localRangeMax = Math.max(value, localRangeMax);
				}
			}
		} else {
			for (int i = 0; i != data.length; i++) {
				for (int j = 0; j != data[0].length; j++) {
					double value = data[i][j];
					localDomainMin = Math.min(value, localDomainMin);
					localDomainMax = Math.max(value, localDomainMax);
				}
			}
		}
	}

	protected double[] getIndMinMaxRange(double[][] data){
		double individualRangeMin = Double.NEGATIVE_INFINITY;
		double individualRangeMax = Double.POSITIVE_INFINITY;

		for(int i = 0; i < data.length; i++){
			for(int cv = 0; cv < data.length; cv++){
				individualRangeMin = Math.min(individualRangeMin,data[i][cv]);
				individualRangeMax = Math.max(individualRangeMax,data[i][cv]);
			}
		}
		return new double[] {individualRangeMin, individualRangeMax};
	}

	protected void getOverallMinMaxValues(double[][] data, double [][] ul, double [][] ll,
				int type, int axis) {
		if (data == null & ul == null & ll == null) {
			return;
		}
		
		if (type == GLOBAL && axis == RANGE) {
			for (int i = 0; i != data.length; i++) {
				for (int j = 0; j != data[0].length; j++) {
					double upvalue = ul[i][j];
					double lowvalue = ll [i][j];
					globalRangeMin = Math.min(lowvalue, globalRangeMin);
					globalRangeMax = Math.max(upvalue, globalRangeMax);
				}
			}
		} else if (type == GLOBAL) {
			for (int i = 0; i != data.length; i++) {
				for (int j = 0; j != data[0].length; j++) {
					double upvalue = ul[i][j];
					double lowvalue = ll [i][j];
					globalDomainMin = Math.min(lowvalue, globalDomainMin);
					globalDomainMax = Math.max(upvalue, globalDomainMax);
				}
			}
		} else if (type == LOCAL && axis == RANGE) {
			for (int i = 0; i != data.length; i++) {
				for (int j = 0; j != data[0].length; j++) {
					double upvalue = ul[i][j];
					double lowvalue = ll [i][j];
					localRangeMin = Math.min(lowvalue, localRangeMin);
					localRangeMax = Math.max(upvalue, localRangeMax);
				}
			}
		} else {
			for (int i = 0; i != data.length; i++) {
				for (int j = 0; j != data[0].length; j++) {
					double upvalue = ul[i][j];
					double lowvalue = ll [i][j];
					localDomainMin = Math.min(lowvalue, localDomainMin);
					localDomainMax = Math.max(upvalue, localDomainMax);
				}
			}
		}
	}

	protected void applyMinMaxValues(JFreeChart chart, int type, int axis) {
		ValueAxis rangeAxis = null;
		ValueAxis domainAxis = null;
		
		Plot plot = chart.getPlot();
		if (plot instanceof CategoryPlot) {
			rangeAxis = ((CategoryPlot) plot).getRangeAxis();
		} else if (plot instanceof XYPlot) {
			rangeAxis = ((XYPlot) plot).getRangeAxis();
			domainAxis = ((XYPlot) plot).getDomainAxis();
		} else {
			return;
		}
		
		// Sets the overall max and min values and adds 5% of white space
		// to make the max and min values more visible.
		double extra; 
		if (type == GLOBAL && axis == RANGE) {
			extra = 0.05* (globalRangeMax - globalRangeMin);
			rangeAxis.setUpperBound(globalRangeMax + extra);
			
			if (!addWhiteSpaceAtBottom()) {
				extra = 0;
			}
			rangeAxis.setLowerBound(globalRangeMin - extra);
		} else if (type == GLOBAL) {
			extra = 0.05 * (globalDomainMax - globalDomainMin);
			domainAxis.setUpperBound(globalDomainMax + extra);
			
			if (!addWhiteSpaceAtBottom()) {
				extra = 0;
			}
			domainAxis.setLowerBound(globalDomainMin - extra);
		} else if (type == LOCAL && axis == RANGE) {
			extra = 0.05 * (localRangeMax - localRangeMin);
			rangeAxis.setUpperBound(localRangeMax + extra);
			
			if (!addWhiteSpaceAtBottom()) {
				extra = 0;
			}
			rangeAxis.setLowerBound(localRangeMin - extra);
		} else {
			extra = 0.05 * (localDomainMax - localDomainMin);
			domainAxis.setUpperBound(localDomainMax + extra);
			
			if (!addWhiteSpaceAtBottom()) {
				extra = 0;
			}
			domainAxis.setLowerBound(localDomainMin - extra);
		}
	}
	
	public void doSaveAs() {
		ImageWriter.doSaveAs(mChartPanel);
	}
	
	protected void doSaveAs(ChartPanel panel){
		ImageWriter.doSaveAs(panel);
	}

	/**
	 * Returns the filename of the currently selected mFileComboBox item.
	 * @return The filename of the currently selected item
	 */
	public String getCurrentFile(){
		return (String) mFileComboBox.getSelectedItem();
	}

	/**
	 * Derived classes should override this function to be
	 * able to filter out useless models.
	 */
	protected boolean ModelIsApplicable(ResultModel model) {
		return true;
	}
}