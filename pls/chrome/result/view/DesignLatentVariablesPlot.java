package pls.chrome.result.view;


import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.event.ActionEvent;

import java.text.DecimalFormat;
import java.util.ArrayList;

import javax.swing.JComboBox;
import javax.swing.JTabbedPane;

import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.labels.StandardCategoryItemLabelGenerator;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.DefaultKeyedValues;
import org.jfree.data.category.CategoryDataset;
import org.jfree.data.general.DatasetUtilities;

import Jama.Matrix;

import java.io.File;
import java.util.Set;
import org.jfree.chart.plot.CategoryPlot;
import pls.shared.GlobalVariablesFunctions;
import pls.chrome.result.controller.observer.Event;
import pls.chrome.result.controller.observer.datachange.*;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.PlsResultModel;
import pls.chrome.result.model.ResultModel;
import pls.shared.MLFuncs;

@SuppressWarnings("serial")
public class DesignLatentVariablesPlot extends AbstractPlot implements DataChangeObserver{

	private double[][] mPercentage;
	private double[][] mProbability;

	private JComboBox mLvComboBox = new JComboBox();
	//private ChartPanel[] mDesignLVChartPanel;

	/*Persistence variables*/
	ChartPersistence mSettings;
	/*End persistence variables*/

	private JTabbedPane mTabs = new JTabbedPane();
	
	public DesignLatentVariablesPlot(String title, GeneralRepository repository) {
		super(title, repository, GlobalVariablesFunctions.PLS);

		mTabs.addMouseListener(new MouseAdapter() {
			@Override
			public void mouseClicked(MouseEvent e) {
				JTabbedPane src = (JTabbedPane) e.getSource();
				int selectedIndex = src.getSelectedIndex();

				//save the selected group for the particular file.
				mSettings.setSelectedTab(getCurrentFile(), selectedIndex);
			}
		});
		
		remove(mChartPanel);
		mFilePanel.add(mLvComboBox);
		add(mTabs, BorderLayout.CENTER);
		mRepository.getPublisher().registerObserver(this);
	}
	
	public double[][] getRangeData(ResultModel model) {
		return model.getDesignLv();
	}

	private void initSettings(){
		mSettings = new ChartPersistence();
		mSettings.addNewFile(getCurrentFile());
		mSettings.setSelectedLV(getCurrentFile(), 0);
	}
	
	/**
	 * Creates the settings object for this plot if it has not been already
	 * created. Adds persistence information for any newly discovered files
	 * (will add panels for each group found and null charts for each lv in that
	 * group). Resets the tabs (redraws them) and resets the lvcombo box. Finally
	 * sets the probability for this plot.
	 * @param fileIndex
	 */
	@Override
	protected void updateLvComboBoxAndTabs(int fileIndex) {
		ResultModel model = mRepository.getGeneral(mResultFilePaths.get(fileIndex) );
		PlsResultModel newModel = (PlsResultModel) model;

		int numLvs = model.getBrainData().getNumLvs();
		String variableType = model.getVariableType();
		
		if(mSettings == null){
			initSettings();
		}
		updateSettings();

		// Adds a dropdown menu for selecting a different CV/LV.
		if (mLvComboBox == null) {
			mLvComboBox = new JComboBox();
		}
		//remove items in the lv combobox in prep for loading items for the newly
		//selected item
		mLvComboBox.removeActionListener(this);
		mLvComboBox.removeAllItems();
		
		for (int i = 0; i < numLvs; i++) {
			mLvComboBox.addItem(variableType + " #" + (i + 1));
		}
		mLvComboBox.addActionListener(this);
		
		// Creates tabs where each tab represents one group.
		int numGroups = model.getNumSubjectList().length;

		/*Add a blank panel for each group*/
		ArrayList<ChartPanel> panels = mSettings.getPanels(getCurrentFile());
		ArrayList<ArrayList<JFreeChart>> charts = mSettings.getCharts(getCurrentFile());

		while(numGroups > panels.size()){
			ChartPanel newChart = new ChartPanel(null);
			panels.add(newChart);
			charts.add(new ArrayList<JFreeChart>());
			
			int group = charts.size() - 1; //last added group
			for(int i = 0; i< numLvs; i++){
				//initialize all the associated charts for this group to null
				charts.get(group).add(null);
			}
		}
			
		if (mTabs == null) { 
			mTabs = new JTabbedPane();
			mTabs.addMouseListener(new MouseAdapter() {
				@Override
				public void mouseClicked(MouseEvent e) {
					JTabbedPane src = (JTabbedPane) e.getSource();
					int selectedIndex = src.getSelectedIndex();

					//save the selected group for the particular file.
					mSettings.setSelectedTab(getCurrentFile(),selectedIndex);
				}
			});
		}

		mTabs.removeAll();
		for (int i = 0; i < numGroups; i++) {
			String cf = getCurrentFile();
			Dimension d = mSettings.getChartPanel(cf, i).getPreferredSize();
			d.height = d.height / 2;
			mSettings.getChartPanel(cf, i).setPreferredSize(d);
			mTabs.addTab("Group " + (i + 1), mSettings.getChartPanel(cf, i));
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


	//This should be called once for each open plot because each plot may be
	//displaying a different file and hence will not have the same local ranges.
	public void calculateLocalRange(int fileIndex){
		//get min values
		localRangeMin = Double.POSITIVE_INFINITY;
		localRangeMax = Double.NEGATIVE_INFINITY;

		//calculate the overall min and max values (over all lvs) for
		//this particular file.
		double[][] designLV = mRangeData.get(fileIndex);
		getMinMaxValues(designLV, LOCAL, RANGE);
	}

	public void makeChart(int fileIndex) {
		calculateLocalRange(fileIndex); //pretty sure this is unnecessary

		//initialize the charts if they have not been initialized already.
		String currentFile = getCurrentFile();
		if(!mSettings.getChartsInitialized(currentFile)){
			initializeCharts(lookupFileIndex(currentFile));
			mSettings.setChartsInitialized(currentFile);
		}

		int selectedLv = mSettings.getSelectedLV(getCurrentFile());
		mLvComboBox.setSelectedIndex(selectedLv);
		makeLvChart(fileIndex, selectedLv);
		mTabs.setSelectedIndex(mSettings.getSelectedTab(getCurrentFile()));
	}

	public void makeLvChart(int fileIndex, int c) {
		ResultModel model = mRepository.getGeneral(mResultFilePaths.get(fileIndex));
		double[][] designLV = mRangeData.get(fileIndex);
		
		ArrayList<String> conditionNames = model.getConditionNames();
		String abbrVariableType = model.getAbbrVariableType().toUpperCase();
		
		int numConditions = conditionNames.size();
		int numGroups = model.getNumSubjectList().length;

		double[] designLVData = MLFuncs.getColumn(designLV, c);
		for (int i = 0; i < numGroups; i++) {
			DefaultKeyedValues designLVKeyedData = new DefaultKeyedValues();
			for (int j = i * numConditions, conditionNumber = 0; j < (i * numConditions)
					+ numConditions; j++, conditionNumber++) {
				designLVKeyedData.addValue(conditionNames.get(conditionNumber),
						designLVData[j]);
			}

			CategoryDataset designLVDataset = DatasetUtilities
					.createCategoryDataset("Value", designLVKeyedData);
			String percent = new DecimalFormat("###.###").format(mPercentage[0][c]);
			JFreeChart designLVChart = mSettings.getChart(getCurrentFile(), i, c);

			if(designLVChart == null){ //the chart has never been graphed before, do so now.
				designLVChart = ChartFactory.createStackedBarChart(
						"Design latent variables for group " + (i + 1)
						+ ", Design " + abbrVariableType + " " + (c + 1) + "\n" + percent
						+ "% crossblock, p < " + mProbability[0][c], "Conditions",
						"Weights", designLVDataset, PlotOrientation.VERTICAL,
						false, true, false);

				designLVChart.getCategoryPlot().getRenderer().setBaseItemLabelsVisible(true);
				designLVChart.getCategoryPlot().getRenderer()
						.setBaseItemLabelGenerator(new StandardCategoryItemLabelGenerator() );
				mSettings.setChart(getCurrentFile(), i, c, designLVChart);
			}
			else{ //we graphed this chart before but we redraw it anyways (the data may have changed)
				CategoryPlot plot = (CategoryPlot) designLVChart.getPlot();
				plot.setDataset(designLVDataset); //this does NOT change the axis.
			}									 //it just changes what is plotted.
			
			mSettings.getChartPanel(getCurrentFile(), i).setChart(designLVChart);
		}
	}

	/**
	 * This function is here so that when we load another file and if the radio
	 * button is selected on the LOCAL/GLOBAL range this newly loaded file gets
	 * its charts initialized with that range. It also creates all the charts.
	 */

	protected void initializeCharts(int fileIndex){
		String file = (String) mFileComboBox.getItemAt(fileIndex);
		int type;

		if(mGlobalButton.isSelected()){
			type = GLOBAL;
		}else if(mLocalButton.isSelected()){
			type = LOCAL;
			calculateLocalRange(fileIndex);
		}else{ type = 2; }

		for(ArrayList<JFreeChart> group : mSettings.getCharts(file)){
			for(int lv = 0; lv < group.size(); lv++){
				if(group.get(lv) == null){
					makeLvChart(lookupFileIndex(file), lv);
					if(type != 2)
						applyMinMaxValues(group.get(lv),type,RANGE);
				}
			}
		}
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
					if(type == GLOBAL){
						applyMinMaxValues(chart, type, RANGE);
					}else if(type == LOCAL){
						applyMinMaxValues(chart,type,RANGE);
					}
					else{
						((CategoryPlot) chart.getPlot()).getRangeAxis().setAutoRange(true);
					}
				}
			}
		}
		//just to be on the safe side, make sure the local range is consistant
		//with the file we are currently working with
		calculateLocalRange(lookupFileIndex(getCurrentFile()));
	}

	/**
	 * Called whenever a file is loaded or removed via the result viewer.
	 * //TODO: Abstract this function.
	 */
	public void updateSettings(){
		//if we have never set settings for this plot it means that the plot
		//while existing, does not actually have any data to plot and is invisible
		//For example this plot will still exist even when we only have an npairs
		//plot loaded and in this case mSettings will be null.

		if(mSettings == null)
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

	@Override
	public void actionPerformed(ActionEvent e) {
		if(e.getSource() == mGlobalButton || e.getSource() == mLocalButton
			|| e.getSource() == mNoneButton){
			resetRanges();
		}
		else if (e.getSource() == mLvComboBox) {
			int file = mFileComboBox.getSelectedIndex();
			int lv = mLvComboBox.getSelectedIndex();
			mSettings.setSelectedLV(getCurrentFile(), lv);
			makeLvChart(file, lv); //redraw the selected chart
		} else {
			super.actionPerformed(e);
		}
	}

	@Override
	public void doSaveAs() {
		ChartPanel chartPanel = (ChartPanel) mTabs.getSelectedComponent();
		doSaveAs(chartPanel);
	}

	@Override
	protected boolean ModelIsApplicable(ResultModel model) {
		PlsResultModel plsModel = (PlsResultModel)model;
		return plsModel.getDesignScores() != null && plsModel.getSProbability() != null;
		//return model.getDesignLv() != null;
	}

        @Override
	public void notify(FlipVolumeEvent e) {}

	@Override
	public void notify(LoadedVolumesEvent e) {
		//fixState();
	}

	@Override
	public void notify(InvertedLvEvent e) {
		if(getCurrentFile() != null){
			int selectedFile = mFileComboBox.getSelectedIndex();
			int selectedLv = mSettings.getSelectedLV(getCurrentFile());
			calculateGlobalRange();
			calculateLocalRange(selectedFile);
			makeLvChart(selectedFile,selectedLv);
		}
	}

	@Override
	public void notify(Event e) {}
}