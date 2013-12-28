package pls.chrome.shared;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import javax.swing.JButton;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTabbedPane;
import javax.swing.JTable;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;
import javax.swing.event.TableModelListener;
import javax.swing.table.DefaultTableModel;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.CategoryDataset;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.general.DatasetUtilities;

import pls.chrome.analysis.NpairsAnalysisFrame;
import pls.chrome.result.view.ResultContrastPanel;
import pls.shared.GlobalVariablesFunctions;
import pls.shared.MLFuncs;

@SuppressWarnings("serial")
public class ContrastPanel extends JPanel {
		
	JTabbedPane tabs = null;
	
	ContrastTablePanel[] cTablePanels = null;
	
	double localRangeMin = Double.POSITIVE_INFINITY;
	double localRangeMax = Double.NEGATIVE_INFINITY;
	
	/**
	 * The constructor that the ResultContrastPanel uses.
	 * @param selectedConditions The conditions that were selected when an 
	 * analysis was run.
	 * @param numGrps The number of groups created for the analysis.
	 * @param resultPanel The ResultContrastpanel 
	 */
	public ContrastPanel(ArrayList<String> selectedConditions, 
						 int numGrps, 
						 ResultContrastPanel resultPanel){
		
		tabs = new JTabbedPane();
		cTablePanels = new ContrastTablePanel[numGrps];
		
    	for(int i = 0; i < numGrps; i++){
		    cTablePanels[i] = new ContrastTablePanel(selectedConditions, 
					 								this, 
					 								resultPanel);
		    tabs.addTab("Group " + (i + 1), cTablePanels[i]);
    	}
	    
	    setLayout(new BorderLayout());
		add(tabs, BorderLayout.CENTER);
	}
	
	/**
	 * This constructor is used by the Analysis frame not the result viewer.
	 * @param sessionProfiles The currently loaded session files.
	 * @param selectedConditions The selected conditions.
	 */
	public ContrastPanel(ArrayList<String[]> sessionProfiles, 
						 ArrayList<Integer> selectedConditions,
						 String contrastFilePath) {
		
		Double[][][] data = null;
		String[] conditionNames;
		List<String> sConditions;
		
		tabs = new JTabbedPane();
		cTablePanels = new ContrastTablePanel[sessionProfiles.size()];
		
		//Get the selected conditions with the assumption that all
    	//groups share the same conditions in the same order and that
    	//each group has the same number of conditions.
    	
    	
    	sConditions = new ArrayList<String>();
    	
    	//Extract the condition names from the session files.
    	conditionNames = NpairsAnalysisFrame.getConditionNamesFromGroups(
    			new Vector<String[]>(sessionProfiles));
    	
    	for(int j = 0; j < selectedConditions.size(); j++){
    		if(selectedConditions.get(j) == 1){
    			sConditions.add(conditionNames[j]);
    		}
    	}
    	
    	if(contrastFilePath != null && !contrastFilePath.equals("")){
    		
    		File contrastFile = new File(contrastFilePath);
    		
    		if(!contrastFile.exists() || !contrastFile.canRead()) {
    			GlobalVariablesFunctions.showErrorMessage(
    					"Could not read/find "
    					+ contrastFilePath + ".");
    		}else{
    		
	    		data = parseContrastFile(contrastFilePath,
	    								 sessionProfiles.size(),
	    								 sConditions.size());
	    		if(data == null){
	    			GlobalVariablesFunctions.showErrorMessage(
	    				"Bad contrast file or incorrect condition selection\n"
						+ contrastFilePath + ".");
	    		}
    		}
    	}
		//for each group add a new graph tab.
		for (int i = 0; i < sessionProfiles.size(); i++) {
	    	
			cTablePanels[i] = new ContrastTablePanel(sConditions, 
		    										 this, 
		    										 null);
		    if(data != null){
		    	cTablePanels[i].setContrastFileData(data[i]);
		    }
	    	tabs.addTab("Group " + (i + 1), cTablePanels[i]);
	    }
		
	    setLayout(new BorderLayout());
		add(tabs, BorderLayout.CENTER);
	}
	
	/**
	 * Check the contrast values before saving them out to a file.
	 * Checks that all values are filled in and that each contrast
	 * adds up to zero.
	 */
	boolean checkContrastAddsUpToZero(){
		if(cTablePanels.length == 0) return false;
		int rows = cTablePanels[0].table.getRowCount();
		int cols = cTablePanels[0].table.getColumnCount();
		double sum = 0;
		boolean nonempty = false;
		boolean encounteredNull = false;
		for(int r = 0; r < rows; r++){
			sum = 0;
			nonempty = false;
			encounteredNull = false;
			
			for(int i = 0; i < cTablePanels.length; i++){
				for(int c = 0; c < cols; c++ ){
					Object e = cTablePanels[i].table.getValueAt(r, c);
					if(e != null){
						
						//A blank entry was previously encountered in this row
						//but this row now contains a value! Rows must either
						//be completely filled or not filled at all.
						
						if(encounteredNull) {
							GlobalVariablesFunctions.showErrorMessage(
						    		"Row " + (r+1) + " is not completely filled.");
									return false;
						}
						
						if(e instanceof String){
							sum += Double.parseDouble((String)e);
						}else{
							sum += (Double) e;
						}
						nonempty = true;
					}
					
					//A blank entry was encountered in this row but we have 
					//previously encountered a filled in value in this row.
					if(e == null && nonempty == true){
						GlobalVariablesFunctions.showErrorMessage(
								"Row " + (r+1) + " is not completely filled.");
						return false;
					}
					
					if(e == null) encounteredNull = true;
				}
			}
			
			if(sum != 0){
				GlobalVariablesFunctions.showErrorMessage(
			    	"Contrast " + (r+1) + " does not add up to zero.\n" +
			    			"Sum is: " + sum);
				return false;
			}
		}
		return true;
	}
	
	/**
	 * Load in contrast values from a file.
	 * @param contrastFilePath The path to the file with the contrast values.
	 * @param ngrps The number of groups contained in the file.
	 * @param nconditions The number of conditions.
	 * @return The parsed contrast values using the following as indices
	 * [group][condition][value].
	 */
	private Double[][][] parseContrastFile(String contrastFilePath, 
										   int ngrps,
										   int nconditions){
		
		File contrastFile = new File(contrastFilePath);
		FileReader reader = null;
		BufferedReader bufread = null;
		Double data[][][] = new Double[ngrps][nconditions][];
		int curgrp = 0;
		int curcond = 0;
		int curcol = 0;
		int colsize = -1;
		String line;
		String[] tokens;
		
		try{
			reader = new FileReader(contrastFile);
			bufread = new BufferedReader(reader);
			line = bufread.readLine();
			
			while( line != null ){
				if(ngrps == curgrp){
					//If the condition above is true then we have already 
					//parsed all n groups but apparently this file still has
					//more data to be parsed. This must be an error unless
					//it is a blank line we have just read in.
					if(line.trim().equals("")){
						line = bufread.readLine();
						continue;
					}
					return null;
				}
				tokens = line.trim().split("\\s+");
				
				//Check that all rows have the same number of columns.
				if(colsize == -1){ 
					colsize = tokens.length; 
				}
				else if(colsize != tokens.length){
					//all conditions must have the same amount of data entered
					//as all other conditions. if a condition has more or less
					//this is an error.
					return null;
				}
				
				data[curgrp][curcond] = new Double[tokens.length];
				for(String value : tokens){
					try{
						Double parsedValue = Double.parseDouble(value);
						data[curgrp][curcond][curcol++] = parsedValue;
					}catch(NumberFormatException e){
						return null;
					}
				}
				
				//curcond is 0-based while nconditions is 1 based. 
				if(curcond+1 == nconditions){ //begin parsing next group
					curgrp++;
					curcond = 0;
					curcol = 0;
				}else{ //beging parsing next condition
					curcol = 0;
					curcond++;
				}
				
				line = bufread.readLine();
			}
			
			//More data was required than was provided. We failed to load in 
			//all the groups.
			if(ngrps != curgrp) return null;
			
		}catch(IOException e){
			try{
				if(reader != null) reader.close();
			}catch(IOException e2){
				return null;
			}
			return null;
		}
		
		try{
			reader.close();
		}catch(IOException e){return null;}
		
		return data;
	}
		
	/**
	 * Fill the contrast panel with contrast information from a loaded file.
	 * @param fileName The filename with contrast information.
	 */
	void load(String fileName){
		Double[][][] data = null;
		int ngrps = cTablePanels.length;
		
		if(ngrps  < 1) return;
		
		int nconditions = cTablePanels[0].getModel().getColumnCount();
		
		if(nconditions < 1) return;
		
		data = parseContrastFile(fileName, ngrps, nconditions);
		
		if(data == null){
			GlobalVariablesFunctions.showErrorMessage(
					"Bad contrast file or incorrect condition selection\n"
					+ fileName + ".");
			return;
		}
		
		for(int i = 0; i < ngrps; i++){
			cTablePanels[i].clearTable();
			cTablePanels[i].setContrastFileData(data[i]);
		}
	}
	
	/**
	 * Erase the table.
	 */
	void clearTable(){
		for(int i = 0; i < cTablePanels.length; i++){
			cTablePanels[i].clearTable();
			cTablePanels[i].repaint();
		}
	}
	
	public void fillTable(double[][] values) {
		if (values == null) {
			return;
		}
		
		int first = 0;
		int last = -1;
		
		for (int i = 0; i < cTablePanels.length; i++) { //the number of groups
			for (int j = 0; j < values.length; j++) { //the number of contrasts
				last = values[j].length/cTablePanels.length;
				for (int k = 0; k < cTablePanels[i].table.getColumnCount(); k++) {
					cTablePanels[i].table.setValueAt(values[j][k + first], j, k);
				}
			}
			first += last;
			
			// Reset rest of the table
			for (int j = values.length; j < cTablePanels[i].table.getRowCount(); j++) {
				for (int k = 0; k < cTablePanels[i].table.getColumnCount(); k++) {
					cTablePanels[i].table.setValueAt("", j, k);
				}
			}
			cTablePanels[i].table.getSelectionModel().setSelectionInterval(0, 0);
		}
	}
}

@SuppressWarnings("serial")
final class ContrastTablePanel extends JPanel {
	
	public final int ROWCOUNT = 25;
	
	private DefaultTableModel model = null;
	
	final JTable table;
	
	private ChartPanel chartPanel;
	
	private ContrastPanel parent;
	
	private ResultContrastPanel resultPanel;

	public ContrastTablePanel(List<String> conditions, 
							  final ContrastPanel parent, 
							  ResultContrastPanel resultPanel) {
		
		CategoryDataset dataset = DatasetUtilities.createCategoryDataset(
							"X",
							"Y", 
							new Number[][]{});
		
		JFreeChart chart = ChartFactory.createStackedBarChart(
								"", 
							  "",
							  "",
							  dataset,
							  PlotOrientation.VERTICAL, 
							  false, false, false);
		
		chartPanel = new ChartPanel(chart, false);
		//resultPanel.mChartPanel = chartPanel;
		
		this.parent = parent;
		this.resultPanel = resultPanel;
		
		table = new JTable();
		model = new NewDefaultTableModel(parent, this, resultPanel);
		
		for(String c : conditions) {
			model.addColumn(c);
		}
		
		for(int i = 0; i < ROWCOUNT; i++) {
			model.addRow(new Object[]{});
		}
		
		table.getSelectionModel().addListSelectionListener(new ListSelectionListener() {
			public void valueChanged(ListSelectionEvent ae) {
				createChart();
			}
		});
		
		table.setModel(model);
		
		table.setPreferredScrollableViewportSize(new Dimension(table.getPreferredScrollableViewportSize().width, 150));
		
		this.setLayout(new BorderLayout());
		add(new JScrollPane(table), BorderLayout.NORTH);
		add(chartPanel, BorderLayout.CENTER);
		
//		JButton stats = new JButton("Stats");
//		stats.addActionListener(new ActionListener(){
//			public void actionPerformed(ActionEvent e){
//				int cc = table.getColumnCount();
//				int rc = table.getRowCount();
//				for(int r = 0; r < rc; r++){
//					for(int c = 0; c < cc; c++){
//						System.out.print(table.getValueAt(r, c) + " ");
//					}
//					System.out.println();
//				}
//				parent.checkContrastAddsUpToZero();
//			}
//		});
//		
//		add(stats,BorderLayout.SOUTH);
	}
	
	DefaultTableModel getModel(){
		return model;
	}
	
	/**
	 * Empty the table.
	 */
	void clearTable(){
		TableModelListener[] listeners = model.getTableModelListeners();
		
		for(TableModelListener listener : listeners){
			model.removeTableModelListener(listener);
		}
		
		for(int i = 0; i < ROWCOUNT; i++){
			model.removeRow(0);
		}
		for(int i = 0; i < ROWCOUNT; i++){
			model.addRow(new Object[]{});
		}
		
		for(TableModelListener listener : listeners){
			model.addTableModelListener(listener);
		}
	}
	/**
	 * Fill in contrast information loaded from a file.
	 * @param data Contrast information for a single group.
	 */
	void setContrastFileData(Double[][] data){
		//We need to transpose the data but unfortunately we need to convert
		//the data from type Double to double then back to Double.
		//The data loaded in is so small this shouldn't represent any 
		//noticeable decrease in performance however.
		double[][] pvalues = new double[data.length][data[0].length];
		Double[][] fixedData;
		
		for(int r = 0; r < data.length; r++){
			for(int c = 0; c < data[r].length; c++){
				pvalues[r][c] = data[r][c];
			}
		}
		
		pvalues = MLFuncs.transpose(pvalues);
		fixedData = new Double[pvalues.length][pvalues[0].length];
		
		for(int r = 0; r < pvalues.length; r++){
			for(int c = 0; c < pvalues[r].length; c++){
				fixedData[r][c] = pvalues[r][c];
			}
		}
		
		for(int r = 0; r < pvalues.length; r++){
			for(int c = 0; c < pvalues[r].length; c++){
				model.setValueAt(fixedData[r][c], r, c);
			}
		}
	}
	
	/**
	 * Reads the values of the currently-selected row in the table and
	 * creates the chart using those values.
	 */
	  
	protected void createChart() {
		DefaultCategoryDataset dataset = new DefaultCategoryDataset();
		for (int i = 0; i < table.getColumnCount(); i++) {
			Object currValue = table.getValueAt(table.getSelectedRow(), i);
			if (currValue == null || currValue.toString().trim().equals("")) {
				continue;
			}
			
			dataset.addValue(new Double(currValue.toString()), "Value", table.getColumnName(i));
		}
		JFreeChart chart = ChartFactory.createStackedBarChart("Contrast " + (table.getSelectedRow() + 1), "", "", dataset, PlotOrientation.VERTICAL, false, false, false);
		
		if (parent.localRangeMin != Double.POSITIVE_INFINITY && parent.localRangeMax != Double.NEGATIVE_INFINITY) {
			
			// Calculates extra space to be shown so the charts will be more visible.
			double extra;
			if (resultPanel != null && resultPanel.mGlobalButton.isSelected()) {
				extra = 0.05 * (resultPanel.globalRangeMax - resultPanel.globalRangeMin);
			} else {
				extra = 0.05 * (parent.localRangeMax - parent.localRangeMin);
			}
		
			// Sets the overall max and min range values to the chart being set here.
			ValueAxis rangeAxis = chart.getCategoryPlot().getRangeAxis();
			
			if (resultPanel != null && resultPanel.mGlobalButton.isSelected()) {
				rangeAxis.setUpperBound(resultPanel.globalRangeMax + extra);
				rangeAxis.setLowerBound(resultPanel.globalRangeMin - extra);
			} else if (resultPanel != null && resultPanel.mLocalButton.isSelected()) {
				rangeAxis.setUpperBound(parent.localRangeMax + extra);
				rangeAxis.setLowerBound(parent.localRangeMin - extra);
			}
		}
		
		chartPanel.setChart(chart);
	}
	
}

@SuppressWarnings("serial")
final class NewDefaultTableModel extends DefaultTableModel {
	
	private ContrastPanel parent;
	private ContrastTablePanel tablePanel;
	private ResultContrastPanel resultPanel;
	
	public NewDefaultTableModel(ContrastPanel parent, 
								ContrastTablePanel panel, 
								ResultContrastPanel resultPanel) {
		super();
		this.parent = parent;
		this.tablePanel = panel;
		this.resultPanel = resultPanel;
	}
	
	@Override
	public void setValueAt(Object value, int row, int column) {
		Double doubleValue;
		// If the values have been loaded from a contrast file, then
		// each value given would be in the form of a Double due to
		// our load() method.
		if (value instanceof Double) {
			doubleValue = (Double) value;
			
			// Determines the overall min and max range values with each value that is added to
			// the table, so the same range values can be applied to all the charts displayed.
			parent.localRangeMin = Math.min(doubleValue, parent.localRangeMin);
			parent.localRangeMax = Math.max(doubleValue, parent.localRangeMax);
			
			// Sets the value in the table.
			super.setValueAt(value, row, column);
			
		// Otherwise, the value would be in the form of a String. This
		// occurs when the user manually adds/sets a new value in the
		// table.
		} else if (value instanceof String) {
			String strValue = (String) value;
			if (!strValue.equals("")) {
				try {
					doubleValue = Double.parseDouble(strValue);
					
					// Sets the value in the table if it has been parsed as a double successfully.
					super.setValueAt(value, row, column);
				
					// Determines the overall min and max range values with each value that is added to
					// the table, so the same range values can be applied to all the charts displayed.
					parent.localRangeMin = Math.min(doubleValue, parent.localRangeMin);
					parent.localRangeMax = Math.max(doubleValue, parent.localRangeMax);
				
					if (resultPanel != null) {
						resultPanel.globalRangeMin = Math.min(doubleValue, resultPanel.globalRangeMin);
						resultPanel.globalRangeMax = Math.max(doubleValue, resultPanel.globalRangeMax);
					}
						
					// Creates a new chart to replace the current one
					// since a new value has been added/set.
					tablePanel.createChart();
				} catch (NumberFormatException e) {
					
					// If an invalid character was given, it is not set.
					GlobalVariablesFunctions.showErrorMessage("Invalid character given: " + strValue);
				}
			}else{
				//Allow the user to null an entry.
				super.setValueAt(null, row, column);
			}
		}
	}
	
}