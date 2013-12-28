package pls.chrome.result.view;

import extern.NpairsBoxAndWhiskerRenderer;
import extern.NpairsBoxWhiskerAxisFix;
import java.awt.BorderLayout;
import java.awt.event.ActionEvent;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.TreeMap;

import javax.swing.JComboBox;
import javax.swing.JTabbedPane;

import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.CategoryAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.labels.BoxAndWhiskerToolTipGenerator;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.statistics.DefaultBoxAndWhiskerCategoryDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.io.File;
import java.util.List;
import java.util.Set;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.Plot;
import pls.shared.GlobalVariablesFunctions;
import pls.chrome.result.controller.observer.Event;
import pls.chrome.result.controller.observer.datachange.DataChangeObserver;
import pls.chrome.result.controller.observer.datachange.FlipVolumeEvent;
import pls.chrome.result.controller.observer.datachange.InvertedLvEvent;
import pls.chrome.result.controller.observer.datachange.LoadedVolumesEvent;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.NPairsResultModel;
import pls.chrome.result.model.ResultModel;
import pls.shared.MLFuncs;

@SuppressWarnings("serial")
public class NpairsCvScoresPlot extends AbstractPlot implements DataChangeObserver {
	
	private static final String FULL_DATA = "Full-data Reference";
	private static final String TRAINING = "Training";
	private static final String TEST = "Test";
	
	private double[][] mCurrentData;
	private double[][] mFullCvScores;
	private double[][] mTrainingCvScores;
	private double[][] mTestCvScores;
	private double[] mClassLabels;
	private ArrayList<String> mClassNames;
	
	private JTabbedPane mTabs;
	private int mNumGroups;
	private JComboBox mSourceList = new JComboBox();
	private JComboBox mCvList = new JComboBox();
	
	private ChartPersistence mSettings;

	/**
	 * Constructor requires the full cv scores, the class labels, the first
	 * split cv scores, the 2nd split cv scores, and the names of each
	 * condition.
	 */
	public NpairsCvScoresPlot(String title, GeneralRepository repository) {
		super(title, repository, GlobalVariablesFunctions.NPAIRS);

		// Add our stuff to this panel
		mFilePanel.add(mCvList);
		mFilePanel.add(mSourceList);

		if (mTabs == null) {
			mTabs = new JTabbedPane();
			if(getCurrentFile() != null){
				ChartPanel plotByObs = mSettings.getChartPanel(getCurrentFile(), 0);
				ChartPanel plotByClass = mSettings.getChartPanel(getCurrentFile(), 1);

				if (mTabs.getComponentCount() < 2) {
					mTabs.add("Plot by Observation", plotByObs);
					mTabs.add("Plot by Class", plotByClass);
				}
			}
			addTabsListener();
		}

		remove(mChartPanel);
		add(mTabs, BorderLayout.CENTER);
		mRepository.getPublisher().registerObserver(this);
	}

	/**
	 * Adds a listener to mTabs which saves the selected tab the user has
	 * chosen.
	 */
	private void addTabsListener(){
		//Register only one listener
		//By default there already exists one to handle selection of the tabs.
		if(mTabs.getMouseListeners().length != 1) return;
		
		mTabs.addMouseListener(new MouseAdapter() {
			@Override
			public void mouseClicked(MouseEvent e) {
				JTabbedPane src = (JTabbedPane) e.getSource();
				int selectedIndex = src.getSelectedIndex();

				//save the selected group for the particular file.
				mSettings.setSelectedTab(getCurrentFile(), selectedIndex);
			}
		});
	}

	public double[][] getRangeData(ResultModel model) {
		NPairsResultModel newModel = (NPairsResultModel) model;
		
		// Retrieves the min and max range global range values of the
		// three data types here instead of combining them into one
		// large single 2-d array to do the same thing. The data
		// does not need to be stored for this class anyway.
		getMinMaxValues(newModel.getCvScores(), GLOBAL, RANGE);
		getMinMaxValues(newModel.getCvScoresTrain(), GLOBAL, RANGE);
		getMinMaxValues(newModel.getCvScoresTest(), GLOBAL, RANGE);
		
		return null;
	}

	/**
	 * Calculates what the range axis should be by determining the max/min
	 * values over all cvs for the particular file in question.
	 * @param fileIndex The index in the filecombobox that points to the file
	 * name we are calculating the local range for.
	 */
	private void calculateLocalRange(int fileIndex){
		NPairsResultModel model = mRepository.getNpairsModel(
				mResultFilePaths.get(fileIndex));

		// Determine the overall max and min local range values
		// of the current data.
		localRangeMin = Double.POSITIVE_INFINITY;
		localRangeMax = Double.NEGATIVE_INFINITY;

		mFullCvScores = model.getCvScores();
		mTrainingCvScores = model.getCvScoresTrain();
		mTestCvScores = model.getCvScoresTest();

		getMinMaxValues(mFullCvScores, LOCAL, RANGE);
		getMinMaxValues(mTrainingCvScores, LOCAL, RANGE);
		getMinMaxValues(mTestCvScores, LOCAL, RANGE);
	}

	public void makeChart(int fileIndex) {
		NPairsResultModel model = mRepository.getNpairsModel(
				                             mResultFilePaths.get(fileIndex));
		String currentFile = getCurrentFile();

		mClassLabels = model.getClassLabels();
		mClassNames = model.getClassNames();
		mNumGroups = model.getNumSubjectList().length;
		mCurrentData = mFullCvScores;

		calculateLocalRange(fileIndex);

		// Create entries for each type of data (if available)
		if (mSourceList == null) {
			mSourceList = new JComboBox();
		} else {
			mSourceList.removeActionListener(this);
			mSourceList.removeAllItems();
		}

		mSourceList.addItem(FULL_DATA);
		if (mTrainingCvScores != null) {
			mSourceList.addItem(TRAINING);
		}
		if (mTestCvScores != null) {
			mSourceList.addItem(TEST);
		}
		
		/*CV list settings*/
		String selectedSource = mSettings.getSelectedSource(currentFile);

		if (mCvList == null) {
			mCvList = new JComboBox();
		} else {
			mCvList.removeActionListener(this);
			mCvList.removeAllItems();
		}

		/*Loads the last selected data source, setting mCurrentData*/
		for(int i = 0; i < mSourceList.getItemCount(); i++){
			if(selectedSource.equals(mSourceList.getItemAt(i))){
				mSourceList.setSelectedIndex(i);
				setCurrentData(selectedSource);
				resetCVList(model,selectedSource);
			}
		}

		int selectedCV = mSettings.getSelectedLV(currentFile);
		mCvList.setSelectedIndex(selectedCV);

		/*This happens the first time a chart for a particular file is made
		  This function is NOT called if this file was the first file that
		  was ever loaded in the result viewer however.*/
		if(!mSettings.getChartsInitialized(currentFile)){
			initializeSettings(currentFile);
			initializeCharts(currentFile);
			mSettings.setChartsInitialized(currentFile);
		}else{
			fillPanels(selectedCV);
			/*I have no idea why this is required and nowhere else, but it is.*/
			mTabs.getComponentAt(0).repaint();
			mTabs.getComponentAt(1).repaint();
		}

		int selectedTab = mSettings.getSelectedTab(currentFile);
		mTabs.setSelectedIndex(selectedTab);

		// add back action listeners now that indexs have been added.
		mCvList.addActionListener(this);
		mSourceList.addActionListener(this);
	}

	/**
	 * Reset the CV list. When a different source is chosen it may have more
	 * or less CVs than the previous source and so we need to redraw the CV
	 * list.
	 * @param model the model the user has selected
	 * @param source the datatype (source) which the user has selected.
	 */
	private void resetCVList(ResultModel model, String source){
		// Create entries in the combo box for each CV
		String abbrVariableType = model.getAbbrVariableType();
		int length = 0;

		if(source.equals(FULL_DATA)){
			length = mFullCvScores[0].length;
		}else if (source.equals(TRAINING)){
			length = mTrainingCvScores[0].length;
		}else if (source.equals(TEST)){
			length = mTestCvScores[0].length;
		}

		for (int i = 0; i < length; ++i) {
			mCvList.addItem(abbrVariableType + " #" + (i + 1) );
		}
	}
	/**
	 * Sets the current data that the ploting code should be plotting.
	 * If the user has selected FULL_DATA, TRAINING, or TEST, the current
	 * data will assigned to mFULLCvScores, mTrainingCvScores, or mTestCvScores
	 * respectively.
	 * @param s the type of data that the user wants plotted.
	 */
	private void setCurrentData(String s){
		if (s.equals(FULL_DATA) ) {
			mCurrentData = mFullCvScores;
		} else if (s.equals(TRAINING) ) {
			mCurrentData = mTrainingCvScores;
		} else if (s.equals(TEST) ) {
			mCurrentData = mTestCvScores;
		}
	}

	@Override
	public void doSaveAs() {
		ChartPanel chartPanel = (ChartPanel) mTabs.getSelectedComponent();
		doSaveAs(chartPanel);
	}

	/**
	 * Charts can be zoomed in and have their axis manually changed but when
	 * a new plot range is selected by clicking one of the radio buttons
	 * all the plots will have their axis reset to what the user clicked on.
	 * All axis persistence will be reset accross all files in otherwords.
	 * @param fileIndex
	 */
	private void resetRanges(){
		int type;
		int fileComboBoxIndex = 0;

		if(mGlobalButton.isSelected()){
			type = GLOBAL;
		}else if(mLocalButton.isSelected()){
			type = LOCAL;
		}else{ type = 2; } //let 2 = a constant for the INDIVIDUAL range.

		/*Reset all plots over all files*/
		for (String file : mSettings.getFiles()){
			fileComboBoxIndex = lookupFileIndex(file);

			//calculate and apply the appropriate local range for
			//the file we are currently looking at.
			calculateLocalRange(fileComboBoxIndex);
			for ( ArrayList<JFreeChart> group : mSettings.getCharts(file) ){
				for ( JFreeChart chart : group){
					Plot plot = chart.getPlot();

					if(plot instanceof XYPlot){
						if(type == GLOBAL){
							applyMinMaxValues(chart, type, RANGE);
							((XYPlot) plot).getDomainAxis().setAutoRange(true);
						}else if(type == LOCAL){
							applyMinMaxValues(chart,type,RANGE);
							((XYPlot) plot).getDomainAxis().setAutoRange(true);
						}else{
							((XYPlot) plot).getRangeAxis().setAutoRange(true);
							((XYPlot) plot).getDomainAxis().setAutoRange(true);
						}
					}else{
						if(type == GLOBAL){
							applyMinMaxValues(chart, type, RANGE);
						}else if(type == LOCAL){
							applyMinMaxValues(chart,type,RANGE);
						}else{
							NpairsBoxWhiskerAxisFix.fixBoxAndWhiskerRange(
									(CategoryPlot) plot);
						}
					}
				}
			}
		}
		//just to be on the safe side, make sure the local range is consistant
		//with the file we are currently working with
		calculateLocalRange(lookupFileIndex(getCurrentFile()));
	}

	/**
	 * Our box and whisker plots have been set up to plot 'far out values'.
	 * Values that JFreeChart does not even consider outliers because they are
	 * so annomalous. Currently, even though they are being drawn the axis of
	 * the plot is not correct so they not displayed. Fix the axis while giving
	 * 5% space at the top of the chart.
	 * @param plot the box and whisker plot to fix.
	 */
	private void fixBoxAndWhiskerRange(CategoryPlot plot){
		DefaultBoxAndWhiskerCategoryDataset dataset;
		ValueAxis rangeAxis = plot.getRangeAxis();

		double max = Double.NEGATIVE_INFINITY;
		double min = Double.POSITIVE_INFINITY;
		double maxRegular;
		double minRegular;
		double extra;

		dataset = (DefaultBoxAndWhiskerCategoryDataset) plot.getDataset();

		//Calculate the max and min over all outliers _including_ 'far outs'.
		//getOutliers() actually returns 'far out' values as well.
		for (int c = 0; c < dataset.getColumnCount(); c++) {
			List<Double> outliers = dataset.getOutliers(0,c);

			for (double out : outliers) {
				max = Math.max(max, out);
				min = Math.min(min, out);
			}
		}

		/* max/min regular represent the max and min values of the plot
		 * (excluding 'far outs'). 'Far outs' are outliers that JFreeChart
		 * considers so far out from any box that they cannot even be considered
		 * outliers and are usually not plotted (except in our case because the
		 * code has been hacked to do so). If all our outliers are below all
		 * the boxes then we want the top of our chart to be the top most point
		 * of any box. The reverse condition applies when all our points are at
		 * the top of all boxes*/

		maxRegular = dataset.getRangeUpperBound(true);
		minRegular = dataset.getRangeLowerBound(true);
		//case #1. All outliers are below upper quartile (q3)
		if(maxRegular > max){
			max = maxRegular;
		}
		
		//case #2. All outliers are above lower quartile (q1)
		if(min > minRegular){
			min = minRegular;
		}

		extra = (max - min) * .05;
		rangeAxis.setUpperBound(max + extra);
		rangeAxis.setLowerBound(min - extra);
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == mCvList || e.getSource() == mSourceList) {
			
			// Get the currently selected CV and data type
			int c = mCvList.getSelectedIndex();
			String s = (String) mSourceList.getSelectedItem();
			boolean differentSource = !s.equals(
					mSettings.getSelectedSource(getCurrentFile()));

			setCurrentData(s); //update data source (full, test, etc).

			//if the selected cv does not exist for the selected source
			//reset the selected cv to 0.
			if(s.equals(FULL_DATA) && (c > mFullCvScores[0].length)) c = 0;
			else if (s.equals(TRAINING) && (c > mTrainingCvScores[0].length))
					c = 0;
			else if ((s.equals(TEST) && (c > mTestCvScores[0].length))) c = 0;

			mSettings.setSelectedLV(getCurrentFile(), c);
			mSettings.setSelectedSource(getCurrentFile(), s);

			/*Reset CV listing on selection of new source*/
			
			if(e.getSource() == mSourceList && differentSource){
				String selectedItem = (String)mFileComboBox.getSelectedItem();
				int fileIndex = itemIndex = lookupFileIndex(selectedItem);
				NPairsResultModel model = mRepository.getNpairsModel(
				                             mResultFilePaths.get(fileIndex));

				mCvList.removeActionListener(this);
				mCvList.removeAllItems();
				resetCVList(model,s);
				mCvList.setSelectedIndex(c);
				mCvList.addActionListener(this);
			}

			// Recreate the panels with the new data
			fillPanels(c);

			//Fix the range so that all outliers are displayed for the box and
			//whisker plot when 'individual range' is selected.

			if(mNoneButton.isSelected()){
				Plot plot = mSettings.getChart(getCurrentFile(), 1,c).getPlot();
				fixBoxAndWhiskerRange((CategoryPlot) plot);
			}

		} else if(e.getSource() == mGlobalButton || e.getSource() == mLocalButton
			|| e.getSource() == mNoneButton){
			resetRanges();
		}else{
			super.actionPerformed(e);
		}
	}

	/**
	 * Called whenever a file is loaded or removed via the result viewer.
	 */
	//@Override

	public void updateSettings(){
		if(mSettings == null) //See DesignLatentVariablesPlot.java
			return;
		
		boolean found = false;
		//By the time a file(s) has been added/removed the resultFiles set
		//will reflect this. In otherwords we can be assured that the recently
		//removed/added file is not there/ there by the time we get here.
		Set<String> resultFiles = getResultFiles();
		
		ArrayList<String> addRemoveBuffer = new ArrayList<String>();

		/*When a file(s) is loaded we need to setup and initialize their
		 * settings.*/
		for(String modelName : resultFiles){

			//unfortunately mSettings stores the relative filenames instead
			//of absolute names stored in the resultFiles set so we need to
			//do this ugly bit.
			String fileName =  modelName.substring(
						modelName.lastIndexOf(File.separator) + 1,
						modelName.length());

			if(!mSettings.containsFile(fileName)){
				addRemoveBuffer.add(fileName);
			}
		}

		//We need to do this due to concurrency issues, if we add or remove
		//a file while simulatenously iterating over it a
		//ConcurrentModificationException may be thrown.

		for(String fileName : addRemoveBuffer){
			mSettings.addNewFile(fileName);
		}

		addRemoveBuffer = new ArrayList<String>();
		/*When a file(s) is removed we need to remove the settings we have
		 * stored for it.*/
		for(String models : mSettings.getFiles()){
			for(String activeModels : resultFiles){

				String fileName = activeModels.substring(
						activeModels.lastIndexOf(File.separator) + 1,
						activeModels.length());

				if (models.equals(fileName)){
					found = true;
					break;
				}
			}
			if(found == false){
				addRemoveBuffer.add(models);
			}
			found = false;
		}

		for (String fileName : addRemoveBuffer){
			mSettings.removeFile(fileName);
		}
	}

	/**
	 * When the file is loaded for the first time we need to draw all its charts
	 * and apply the particular range that has been selected via the radio
	 * button.
	 * @param currentFile The current file we are drawing the charts for.
	 */
	private void initializeCharts(String currentFile){
		NPairsResultModel model = mRepository.getNpairsModel(
				mResultFilePaths.get(lookupFileIndex(currentFile)));

		JFreeChart chart;

		mFullCvScores = model.getCvScores();
		mClassLabels = model.getClassLabels();
		mTrainingCvScores = model.getCvScoresTrain();
		mTestCvScores = model.getCvScoresTest();
		mClassNames = model.getClassNames();
		mNumGroups = model.getNumSubjectList().length;
		mCurrentData = mFullCvScores;

		int numCVs = mFullCvScores[0].length;

		for(int group = 0; group < 2; group++){
			for(int cv=0; cv < numCVs; cv++){
				fillPanels(cv);
				chart = mSettings.getChart(currentFile, group, cv);
				
				if(mGlobalButton.isSelected()){
					applyMinMaxValues(chart,GLOBAL,RANGE);
				}else if(mLocalButton.isSelected()){
					calculateLocalRange(lookupFileIndex(currentFile));
					applyMinMaxValues(chart,GLOBAL,RANGE);
				}else if(chart.getPlot() instanceof CategoryPlot){
					fixBoxAndWhiskerRange((CategoryPlot) chart.getPlot());
				}
			}
		}
		//Show the first CV for each chart by default.
		ChartPanel plotByObs = mSettings.getChartPanel(getCurrentFile(), 0);
		ChartPanel plotByClass = mSettings.getChartPanel(getCurrentFile(), 1);
		chart = mSettings.getChart(currentFile,0,0);
		plotByObs.setChart(chart);
		chart = mSettings.getChart(currentFile, 1, 0);
		plotByClass.setChart(chart);

	}

	/**
	 * Initialization settings particular to this plot type, code is executed
	 * when a file is first loaded. It initializes the settings datastructures
	 * for a particular file.
	 * @param filename The file to initialize
	 */
	private void initializeSettings(String filename){
		int fileIndex = lookupFileIndex(filename);

		NPairsResultModel model;
		int numCv;

		ArrayList<ChartPanel> panels;
		ArrayList<ArrayList<JFreeChart>> charts;

		model = mRepository.getNpairsModel(mResultFilePaths.get(fileIndex));
		mFullCvScores = model.getCvScores();
		mTrainingCvScores = model.getCvScoresTrain();
		mTestCvScores = model.getCvScoresTest();
		
		panels = mSettings.getPanels(filename);
		panels.add(new ChartPanel(null));
		panels.add(new ChartPanel(null));

		charts = mSettings.getCharts(filename);

		//Add an arraylist for each group of charts
		//Group 1) Charts belonging to plot by observation
		//Group 2) Charts belonging to plot by class.
		
		charts.add(new ArrayList<JFreeChart>()); //group 1
		charts.add(new ArrayList<JFreeChart>()); //group 2

		//Always allocate to as many charts as possible cvs;
		numCv = Math.max(mFullCvScores[0].length, mTrainingCvScores[0].length);
		numCv = Math.max(numCv, mTestCvScores[0].length);
		//initialize the charts to null
		for(int i = 0; i < numCv; i++){
			charts.get(0).add(null);
			charts.get(1).add(null);
		}
	//selected source, lv, tab are set from the mSettings constructor.
	//specifically during the fixState() function call if this is a new file that
	//has been added when we already previously opened one.
	//otherwise particular init will call this function if this is the
	//first time any file has ever been loaded for this plot.
	}
	
	@Override
	/** Initialization settings particular to this plot, code is executed when
	 *  ever file is added or deleted. Code is also executed for the first file
	 *  ever loaded.
	 */
	protected void particularInit(){
		String currentFile = getCurrentFile();
		if(mSettings == null){
			mSettings = new ChartPersistence();
			mSettings.addNewFile(currentFile);
			initializeSettings(currentFile);
			initializeCharts(currentFile);
			mSettings.setChartsInitialized(currentFile);
		}
		updateSettings();
	}

	// Creates the two types of plots and makes them viewable
	private void fillPanels(int selectedCv) {
		double[] cvScores = MLFuncs.getColumn(mCurrentData, selectedCv);

		ChartPanel plotByObs = mSettings.getChartPanel(getCurrentFile(), 0);
		ChartPanel plotByClass = mSettings.getChartPanel(getCurrentFile(), 1);

		JFreeChart chartByObs = createPlotByObservation(cvScores, selectedCv);
		JFreeChart chartByClass = createPlotByClass(cvScores, selectedCv);

		plotByObs.setChart(chartByObs);
		plotByClass.setChart(chartByClass);
		
		if (mTabs == null) {
			mTabs = new JTabbedPane();
			addTabsListener();
		}
        
		if (mTabs.getComponentCount() < 2) {
			mTabs.add("Plot by Observation", plotByObs);
			mTabs.add("Plot by Class", plotByClass);
		}else{
			mTabs.setComponentAt(0, plotByObs);
			mTabs.setComponentAt(1, plotByClass);
		}
	}
        
	/**
	 * The method to create the plot by observation xy graph for the CV scores.
	 * @param data the data to be plotted (Y-axis data).
	 * @param cvNum the cv number that corresponds to the data.
	 * @return Retuns an xy plot of the provided data corresponding to the input cvNum.
	 */
	private JFreeChart createPlotByObservation(double[] data, int cvNum) {
		XYSeries obsSeries = new XYSeries("Hello");
                
		for (int i = 0; i < data.length; ++i) {
			obsSeries.add(i, data[i]);
		}
		
		XYSeriesCollection mySeries = new XYSeriesCollection();
		mySeries.addSeries(obsSeries);
		
		JFreeChart cvScoresPlotByObs = mSettings.getChart(getCurrentFile(), 0, cvNum);
		
		if(cvScoresPlotByObs == null){
			cvScoresPlotByObs = EnhancedXY.createSpecialXYLineChart("CV Scores",
					"Observation (Input Volume) No.", "Canonical Scores Dim " + (cvNum + 1),
					mySeries, PlotOrientation.VERTICAL, false, false, false);

			XYPlotChild plot = (XYPlotChild) cvScoresPlotByObs.getPlot();
			plot.setVerticalLines(getVerticalLineValues());
		    plot.setRangeZeroBaselineVisible(true);
			plot.setRangeZeroBaselineStroke(new BasicStroke(1));
			plot.setRangeZeroBaselinePaint(Color.blue);

			mSettings.setChart(getCurrentFile(), 0, cvNum, cvScoresPlotByObs);
		}else{ //set data axis
			XYPlot plot = cvScoresPlotByObs.getXYPlot();
			plot.setDataset(mySeries);
		}

		return cvScoresPlotByObs;
	}

	private JFreeChart createPlotByClass(double[] data, int cvNum) {
		TreeMap<Integer, ArrayList<Double>> categorizedData = new TreeMap<Integer, ArrayList<Double>>();

		/*classlabels are tied to classdata. we can tell which data item
		 * belongs to which classlabel by going through both arrays.
		 * a set of data items belongs to the same set if they are tied to the same
		 * classlabel number. this block creates the set mapping in a treemap*/
		for (int i = 0; i < mClassLabels.length; ++i) {
			ArrayList<Double> classData;
			Integer classLabel = (int) mClassLabels[i];
			if (!categorizedData.containsKey(classLabel)) {
				classData = new ArrayList<Double>();
				categorizedData.put(classLabel, classData);
			} else {
				classData = categorizedData.get(classLabel);
			}

			classData.add(data[i]);
		}

		DefaultBoxAndWhiskerCategoryDataset classDataset = new DefaultBoxAndWhiskerCategoryDataset();
		
		Iterator<Integer> classLabels = categorizedData.keySet().iterator();
		int i = 0;
		while (classLabels.hasNext()) {
			Integer classLabel = classLabels.next();
			classDataset.add(categorizedData.get(classLabel), 0, mClassNames.get(i));
			i++;
		}
		
		JFreeChart cvScoresPlotByClass = mSettings.getChart(getCurrentFile(), 1, cvNum);

		if(cvScoresPlotByClass == null){
			// Create our own box and whisker plot
			CategoryAxis categoryAxis = new CategoryAxis("Condition");
			NumberAxis valueAxis = new NumberAxis("Canonical Scores Dim " + (cvNum + 1) );
			valueAxis.setAutoRangeIncludesZero(false);

			NpairsBoxAndWhiskerRenderer renderer = new NpairsBoxAndWhiskerRenderer();
			renderer.setBaseToolTipGenerator(new BoxAndWhiskerToolTipGenerator());
			
			CategoryPlot plot = new CategoryPlot(classDataset, categoryAxis, valueAxis, renderer);
		    plot.setRangeZeroBaselineVisible(true);
			plot.setRangeZeroBaselineStroke(new BasicStroke(1));
			plot.setRangeZeroBaselinePaint(Color.blue);

			cvScoresPlotByClass = new JFreeChart("CV Scores", JFreeChart.DEFAULT_TITLE_FONT, plot, false);
			mSettings.setChart(getCurrentFile(), 1, cvNum, cvScoresPlotByClass);
		}else{
			CategoryPlot plot = cvScoresPlotByClass.getCategoryPlot();
			plot.setDataset(classDataset);
		}
		return cvScoresPlotByClass;
	}

	/**
	 * Determine the dividing lines between subjects if any.
	 * These lines are plotted vertically on the observation plots to
	 * help us differentiate between subjects.
	 * @return array of domain coordinates where the vertical lines will be
	 * plotted.
	 */
	private double[] getVerticalLineValues() {
		int fileIndex = lookupFileIndex(getCurrentFile());

		NPairsResultModel model = mRepository.getNpairsModel(
				mResultFilePaths.get(fileIndex));

		double[] subjLabels = model.getSplitObjLabels();

		if(subjLabels == null)
			return null; //Only one subject? This situation is possible.
		
		double[] retval;
		double lastLabel;

		ArrayList<Double> lineMarkLocations = new ArrayList<Double>(10);

		//The number of subjects is small so this shouldn't be too inefficient
		lastLabel = subjLabels[0];
		for(int z = 0; z < subjLabels.length; z++){
			if(lastLabel != subjLabels[z]){
				lastLabel = subjLabels[z];
				lineMarkLocations.add(new Double(z));
			}
		}
		
		retval = new double[lineMarkLocations.size()];
		for(int z = 0; z < lineMarkLocations.size(); z++){
			retval[z] = lineMarkLocations.get(z).doubleValue();
		}
		return retval;
	}

	public boolean addWhiteSpaceAtBottom() {
		return true;
	}

	@Override
	public void notify(FlipVolumeEvent e) {}

	@Override
	public void notify(LoadedVolumesEvent e) {}

	@Override
	public void notify(InvertedLvEvent e) {
		if(getCurrentFile() != null){
			int selectedFile = mFileComboBox.getSelectedIndex();
			calculateGlobalRange();
			makeChart(selectedFile);
		}
	}

	@Override
	public void notify(Event e) {}
}