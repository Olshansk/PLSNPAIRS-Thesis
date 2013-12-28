
package pls.chrome.result.view;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;

/**
 * This class is used to maintain persistence settings for a particular plot.
 * This class actually stores the charts for the plot and the plot can
 * then modify the charts when it needs to. It is a prototype class that should
 * eventually be modified into an abstract class where each plot type will
 * create their own persistence class by extending this one.
 */
public class ChartPersistence {
	private Map<String,Info> models = new HashMap<String,Info>();

	/**
	 * Registers a new file to keep persistence settings for.
	 * @param filename The filename in question.
	 */
	protected void addNewFile(String filename){
		//I added this check to make sure that when we add a new file,
		//if we already have persistence for that file it is not overwritten.
		//This check may be unnecessary.
		if(!models.containsKey(filename)){
			models.put(filename, new Info());
		}
	}

	/**
	 * Checks whether a file has been previously registered
	 * @param filename The filename in question.
	 * @return true or false depending on whether the file has been registered
	 * or not.
	 */
	protected boolean containsFile(String filename){
		return models.containsKey(filename);
	}

	/**
	 * Gets a list of the files that this class is currently maintaining
	 * persistence for.
	 * @return Set<String> A set of filenames
	 */
	protected Set<String> getFiles(){
		return models.keySet();
	}

	/**
	 * Removes persistence settings for a particular file, tells the class to
	 * stop maintaining persistence settings for a file.
	 * @param filename The filename in question.
	 */
	protected void removeFile(String filename){
		models.remove(filename);
	}

	/**
	 * Retrieves a chart belonging to a particular file, group (or tab), and
	 * lv.
	 * @param filename the file in question.
	 * @param tab int number defining the group or tab for which the chart
	 * belongs.
	 * @param lv int number defining the lv
	 * @return the chart belonging to the tuple (file,group,lv)
	 */
	protected JFreeChart getChart(String filename, int tab, int lv){
		Info modelInfo = models.get(filename);
		JFreeChart retval = modelInfo.charts.get(tab).get(lv);
		return retval;
	}

	/**
	 * For charts without lv numbers use an lv number of 0 as the default. 
	 * @param filename the file in question.
	 * @param tab int number defining the group or tab for which the chart
	 * belongs.
	 * @return the chart belonging to the tuple (file,tab)
	 */
	protected JFreeChart getChart(String filename, int tab){
		return getChart(filename, tab, 0);
	}

	/**
	 * Like getchart, but sets a chart at the specified position instead.
	 * @param filename the file in question.
	 * @param tab int number defining the group or tab for where we want the
	 * chart to belong.
	 * @param lv int number defining the lv
	 * @param chart the chart to set.
	 */
	protected void setChart(String filename, int tab, int lv, JFreeChart chart){
		Info modelInfo = models.get(filename);
		modelInfo.charts.get(tab).set(lv, chart);
	}

	/**
	 * For charts without lv numbers use an lv number of 0 as the default.
	 * @param filename the file in question.
	 * @param tab int number defining the group or tab for where we want the
	 * chart to belong.
	 * @param chart the chart to set.
	 */
	protected void setChart(String filename, int tab, JFreeChart chart){
		setChart(filename, tab, 0, chart);
	}

	/**
	 * Retrieves the specified panel for which charts are contained in.
	 * @param filename The file we are looking at.
	 * @param tab The tab in which the panel belongs.
	 * @return The specified panel.
	 */
	protected ChartPanel getChartPanel(String filename, int tab){
		Info modelInfo = models.get(filename);
		ChartPanel retval = null;

		if(modelInfo != null){
			retval = modelInfo.panels.get(tab);
		}
		return retval;
	}

	/**
	 * Returns all the charts that correspond to the given filename.
	 * @param filename The specified file for which we want to retrieve its
	 * charts.
	 * @return ArrayList<ArrayList<JFreeChart>> the charts belonging to the
	 * specified file.
	 */
	protected ArrayList<ArrayList<JFreeChart>> getCharts(String filename){
		Info modelInfo = models.get(filename);

		if(modelInfo != null){
			return modelInfo.charts;
		}
		return null;
	}

	/**
	 * Returns all the panels that correspond to the given filename.
	 * @param filename The specified file for which we want to retrieve its
	 * panels.
	 * @return ArrayList<ChartPanel> the panels belonging to the specified file.
	 */
	protected ArrayList<ChartPanel> getPanels(String filename){
		Info modelInfo = models.get(filename);

		if(modelInfo != null){
			return modelInfo.panels;
		}
		return null;
	}

	/**
	 * Saves the currently selected lv for the specified file.
	 * @param filename the file for which we want to save its selectedLV.
	 * @param lv the lv in question.
	 */
	protected void setSelectedLV(String filename,int lv){
		Info modelInfo = models.get(filename);

		if(modelInfo != null){
			modelInfo.selectedLV = lv;
		}
	}

	/**
	 * Retrieves the currently selected lv for the specified file.
	 * @param filename the file for which we want to retrieve its selectedLV.
	 * @return the currently selected lv.
	 */
	protected int getSelectedLV(String filename){
		Info modelInfo = models.get(filename);

		return modelInfo.selectedLV;
	}

	protected void setSelectedTab(String filename, int tab){
		Info modelInfo = models.get(filename);

		if(modelInfo != null){
			modelInfo.selectedTab = tab;
		}
	}

	protected int getSelectedTab(String filename){
		Info modelInfo = models.get(filename);
		return modelInfo.selectedTab;
	}

	protected void setChartsInitialized(String filename){
		Info modelInfo = models.get(filename);

		modelInfo.chartsInitialized = true;
	}

	protected boolean getChartsInitialized(String filename){
		Info modelInfo = models.get(filename);
		return modelInfo.chartsInitialized;
	}

	/*Begin npairscvscores particular settings*/
	protected String getSelectedSource(String filename){
		Info modelInfo = models.get(filename);
		return modelInfo.selectedSource;
	}

	protected void setSelectedSource(String filename, String source){
		Info modelInfo = models.get(filename);
		modelInfo.selectedSource = source;
	}

	/**
	 * Inner class that contains the datastructures that house the persistence
	 * data for a particular file.
	 */
	private class Info {

		public Info(){
			chartsInitialized = false;
			selectedLV = 0;
			selectedTab = 0;
			//default value that is always present "Full-data Reference"
			selectedSource = "Full-data Reference"; 
			panels = new ArrayList<ChartPanel>();
			charts = new ArrayList<ArrayList<JFreeChart>>();
		}

		public int selectedLV; //currently selected lv/cv
		public int selectedTab; //currently selected tab
		public String selectedSource; //npairscvscores (current selected source)
		//has this file's persistence data been initialized?
		public boolean chartsInitialized;
		public ArrayList<ChartPanel> panels; //The panels belonging to the file.
		public ArrayList<ArrayList<JFreeChart>> charts; //The charts belonging..
	}
}
