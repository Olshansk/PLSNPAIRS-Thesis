package pls.chrome.result.view;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import javax.swing.JComboBox;
import javax.swing.JTabbedPane;

import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.CategoryAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.labels.StandardCategoryItemLabelGenerator;
import org.jfree.chart.plot.CategoryPlot;

import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.PlsResultModel;
import pls.chrome.result.model.ResultModel;
import pls.shared.GlobalVariablesFunctions;
import pls.shared.MLFuncs;
import extern.AsymmetricStatisticalBarRenderer;
import extern.AsymmetricStatisticalCategoryDataset;


@SuppressWarnings("serial")
public class TaskPLSwithCIPlot extends AbstractPlot implements MouseListener{
	
	private JComboBox mLvComboBox = new JComboBox();
	private ChartPanel[] mTaskChartPanel;
	private JTabbedPane mTabs = new JTabbedPane();
	private ChartPersistence persistence;

	public TaskPLSwithCIPlot(String title, GeneralRepository repository) {
		super(title, repository, GlobalVariablesFunctions.PLS);
		
		remove(mChartPanel);
		mFilePanel.add(mLvComboBox);
		add(mTabs, BorderLayout.CENTER);

}
	@Override
	public double[][] getRangeData(ResultModel model) {
		return ((PlsResultModel) model).getOrigUsc();
	}
	
	@Override
	protected void updateLvComboBoxAndTabs(int fileIndex) {
		ResultModel model = mRepository.getGeneral(mResultFilePaths.get(fileIndex) );
		int numLvs = model.getBrainData().getNumLvs();
		String variableType = model.getVariableType();
		
		// Adds a dropdown menu for selecting a different CV/LV.
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
		int numGroups = model.getNumSubjectList().length;

		mTaskChartPanel = new ChartPanel[numGroups];
		if (mTabs == null) {
			mTabs = new JTabbedPane();
		}else{
			mTabs.removeMouseListener(this);
			mTabs.removeAll();
		}
		
		for (int i = 0; i < numGroups; i++) {
			mTaskChartPanel[i] = new ChartPanel(null);
			Dimension d = mTaskChartPanel[i].getPreferredSize();
			d.height = d.height / 2;
			mTaskChartPanel[i].setPreferredSize(d);
			mTabs.addTab("Group " + (i + 1), mTaskChartPanel[i]);
		}
		mTabs.addMouseListener(this);
	}
	
	@Override
	public void makeChart(int fileIndex) {
		localRangeMin = Double.POSITIVE_INFINITY;
		localRangeMax = Double.NEGATIVE_INFINITY;
		
		// Retrieves the overall min and max values for the data
		// of the model of the given file index.
		double[][] OrigUsc = mRangeData.get(fileIndex);
		PlsResultModel data  = mRepository.getPlsModel(mResultFilePaths.get(fileIndex) );;
		double [][] ulUsc = data.getUlUsc();
		double [][] llUsc = data.getllUsc();
		
		getOverallMinMaxValues(OrigUsc, ulUsc, llUsc, LOCAL, RANGE);
		getOverallMinMaxValues(OrigUsc, ulUsc, llUsc, GLOBAL, RANGE);
		
		// Uses the first CV/LV as the default selected CV/LV.
		makeLvChart(fileIndex);
	}
	
	public void makeLvChart(int fileIndex) {
	
	if(persistence == null){
		persistence = new ChartPersistence();
	}
	
	String fileName = mResultFilePaths.get(fileIndex);
	ResultModel model = mRepository.getGeneral(fileName);
	double[][] origUsc = mRangeData.get(fileIndex);
		
	PlsResultModel data  = mRepository.getPlsModel(fileName);
	double [][] ulUsc = data.getUlUsc();
	double [][] llUsc = data.getllUsc();
		
	ArrayList<String> conditionNames = model.getConditionNames();
	String abbrVariableType = model.getAbbrVariableType().toUpperCase();
		
	int numConditions = conditionNames.size();
	int numGroups = model.getNumSubjectList().length;
	int numLvs = model.getBrainData().getNumLvs();
	int lv = persistence.containsFile(fileName) ? 
			 persistence.getSelectedLV(fileName) : 0;
			 
	double[] origUscData = MLFuncs.getColumn(origUsc, lv);
	double[] ulUscData = MLFuncs.getColumn(ulUsc, lv);
	double[] llUscData = MLFuncs.getColumn(llUsc, lv);
	
	ArrayList<ArrayList<JFreeChart>> charts;
	
	//Setup persistence information for this file, return true if this is
	//the first time we are setting up persistence for this file.
	if(!setupPersistence(fileName,numGroups,numLvs)){
		charts = persistence.getCharts(fileName);
		int selectedTab = persistence.getSelectedTab(fileName); //?
		
		//Check if we have persistence information for this lv for this file.
		//If we do, display these charts but update their ranges to what has 
		//been selected.
		
		if(persistence.getChart(fileName, selectedTab, lv) != null){
			applyRanges(charts,lv);
			return;
		}
	}
	//
	
	charts = persistence.getCharts(fileName);
	
	//For each group create a chart for the specified lv.
	for (int i = 0; i < numGroups; i++) {
		AsymmetricStatisticalCategoryDataset ds = new AsymmetricStatisticalCategoryDataset();
		for (int j = i * numConditions, conditionNumber = 0; j < (i * numConditions)
				+ numConditions; j++, conditionNumber++) {
			Number orig = origUscData[j];
			Number ul = ulUscData[j];
			Number ll = llUscData[j];
			ds.add(orig, ul, ll, "values", conditionNames.get(conditionNumber));
		}
	
		CategoryAxis xAxis = new CategoryAxis("Conditions");
		ValueAxis yAxis = new NumberAxis("Brain Scores");
			
		AsymmetricStatisticalBarRenderer renderer = new AsymmetricStatisticalBarRenderer();
		CategoryPlot plot = new CategoryPlot(ds, xAxis, yAxis, renderer);
		 
		JFreeChart origUscChart = new JFreeChart("Brain Scores with CI for group " + (i + 1)
					+ ", " + abbrVariableType + " " + (lv + 1), 
					plot);
	  
		origUscChart.getCategoryPlot().getRenderer().setBaseItemLabelsVisible(true);
		origUscChart.getCategoryPlot().getRenderer().setBaseItemLabelGenerator(new StandardCategoryItemLabelGenerator());
			
		if (mGlobalButton.isSelected()) {
			applyMinMaxValues(origUscChart, GLOBAL, RANGE);
		} else if (mLocalButton.isSelected()){
			applyMinMaxValues(origUscChart, LOCAL, RANGE);
		}
		
		charts.get(i).set(lv, origUscChart);	
		mTaskChartPanel[i].setChart(origUscChart);
		
	}
}
	
	@Override
	public void actionPerformed(ActionEvent e) {
		if(e.getSource() == mLvComboBox){
			int file = mFileComboBox.getSelectedIndex();
			int lv = mLvComboBox.getSelectedIndex();
			String fileName = mResultFilePaths.get(file);
			
			persistence.setSelectedLV(fileName, lv);
			makeLvChart(file);
		}
		else if (e.getSource() == mGlobalButton 
				|| e.getSource() == mLocalButton 
				|| e.getSource() == mNoneButton) {
			
			int file = mFileComboBox.getSelectedIndex();
			makeLvChart(file);
		} else {
			super.actionPerformed(e);
		}
	}
	
	/**
	 * Save the chart as an image file.
	 */
	public void doSaveAs() {
		ChartPanel chartPanel = (ChartPanel) mTabs.getSelectedComponent();
		doSaveAs(chartPanel);
	}
	
	@Override
	protected boolean ModelIsApplicable(ResultModel model) {
		PlsResultModel plsModel = (PlsResultModel)model;
		return plsModel.getOrigUsc() != null &&
		plsModel.getUlUsc() != null &&
		plsModel.getUlUsc() != null;
	}
	
	/**
	 * Remove persistence for files that are no longer loaded.
	 * Ideally it would be nice to call this function only when
	 * a file is removed but instead I have decided to let this
	 * function call be made whenever initialize() is called.
	 * This means that not only is this function called when a
	 * file is added/removed but also whenever the user hits
	 * any widget that ends up calling makeLVChart (file selection,
	 * lv selection etc). While less efficient, its cleaner since I don't
	 * need to register this object with a publisher.  
	 */
	private void cleanupPersistence(){
		boolean found = false;
		Set<String> pFiles = persistence.getFiles();
		List<String> removal = new ArrayList<String>(pFiles.size());
		
		for(String file : pFiles){
			for(String loadedFile : mResultFilePaths){
				if(file.equals(loadedFile)) found = true;
			}
			if(!found) removal.add(file);
			found = false;
		}
		
		for(String removeFile : removal){
			persistence.removeFile(removeFile);
		}
	}
	
	/**
	 * Set up any persistence information for the selected file. This means 
	 * setting the previously selected lv, and tab for a file or if the file
	 * has never been initialized setting the lv to the first lv and the tab
	 * to the first tab. 
	 * @param fileName the file in question
	 * @param numGroups the number of groups that a file has
	 * @param numLvs the number of lvs that a file has
	 * @return true if this file has never been initialized, false otherwise.
	 */
	private boolean setupPersistence(String fileName, int numGroups, int numLvs){
		
		cleanupPersistence();
		
		if(!persistence.containsFile(fileName)){
			ArrayList<ArrayList<JFreeChart>> charts;
			
			persistence.addNewFile(fileName);
			persistence.setSelectedLV(fileName, 0);
			persistence.setSelectedTab(fileName, 0);
			charts = persistence.getCharts(fileName);
	
			setLVandTabIndex(0,0);
						
			//For each tab
			for(int i = 0; i < numGroups; i++){
				//create as many spaces for potential charts as there are lvs.
				ArrayList<JFreeChart> lvCharts = new ArrayList<JFreeChart>(numLvs);
				for(int j = 0; j < numLvs; j++){
					lvCharts.add(null);
				}
				charts.add(lvCharts);
			}
			return true;
		}
		
		setLVandTabIndex(persistence.getSelectedLV(fileName),
						persistence.getSelectedTab(fileName));
		
		return false;
	}
	
	private void setLVandTabIndex(int lv, int tab){
		mLvComboBox.removeActionListener(this);
		mLvComboBox.setSelectedIndex(lv);
		mLvComboBox.addActionListener(this);
		mTabs.removeMouseListener(this);
		mTabs.setSelectedIndex(tab);
		mTabs.addMouseListener(this);
	}
	
	/**
	 * Apply the currently selected range to the charts.
	 * @param charts the charts of the currently selected file.
	 * @param lv the lv number which tells us the charts across all groups for
	 * that lv to change the range for. 
	 */
	private void applyRanges(ArrayList<ArrayList<JFreeChart>> charts, int lv){
		int i = 0;
		for(ArrayList<JFreeChart> groups : charts){
			JFreeChart chart = groups.get(lv);
			if (mGlobalButton.isSelected()) {
				
				applyMinMaxValues(chart, GLOBAL, RANGE);
			} else if (mLocalButton.isSelected()){
				
				applyMinMaxValues(chart, LOCAL, RANGE);
			} else{
				chart.getCategoryPlot().getRangeAxis().setAutoRange(true);
			}
			mTaskChartPanel[i].setChart(chart); 
			i++;
		}
	}
	
	@Override
	public void mouseClicked(MouseEvent arg0) {
		String file = mResultFilePaths.get(lookupFileIndex(getCurrentFile()));
		persistence.setSelectedTab(file,mTabs.getSelectedIndex());
	}
	
	@Override
	public void mouseEntered(MouseEvent arg0) {
		// TODO Auto-generated method stub
		
	}
	@Override
	public void mouseExited(MouseEvent arg0) {
		// TODO Auto-generated method stub
		
	}
	@Override
	public void mousePressed(MouseEvent arg0) {
		// TODO Auto-generated method stub
		
	}
	@Override
	public void mouseReleased(MouseEvent arg0) {
		// TODO Auto-generated method stub
		
	}


}