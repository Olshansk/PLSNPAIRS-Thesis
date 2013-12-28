package pls.chrome.shared;

import java.awt.Component;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import javax.imageio.ImageIO;
import javax.swing.JFileChooser;
import javax.swing.filechooser.FileNameExtensionFilter;

import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.XYPlot;

import pls.chrome.result.blvplot.BrainInfoPane;
import pls.chrome.result.view.AbstractFunctionalPlot;
import pls.chrome.result.view.BrainBehavScoresPlot;
import pls.chrome.result.view.scatterplot.EnhancedScatterPlot;
import pls.shared.GlobalVariablesFunctions;

/**
 * Class for saving plots as image files
 */
public class ImageWriter {
	
	private static JFileChooser chooser = getFileChooser();
	
	/**
	 * Initializer for the save file dialog box.
	 * @return a new save file dialog box.
	 */
	private final static JFileChooser getFileChooser() {
		
		//Retrieves the different image formats the plots can be saved as first.
		String[] formatNames = ImageIO.getWriterFormatNames();
		
		ArrayList<String> filteredNames = new ArrayList<String>();
		for (String s : formatNames) {
			
			//Currently we only allow the png and gif formats.
			String uppercase = s.toUpperCase();
			if(uppercase.equals("PNG") && !filteredNames.contains("PNG")){
				filteredNames.add(uppercase);
			}
			if(uppercase.equals("GIF") && !filteredNames.contains("GIF")){
				filteredNames.add(uppercase);
			}
			/*if (!filteredNames.contains(uppercase)) {
				filteredNames.add(uppercase);
			}*/
		}
		
		JFileChooser chooser = new JFileChooser(".");
		for (String s : filteredNames) {
			chooser.addChoosableFileFilter(new FileNameExtensionFilter(s, s));
		}
		
		return chooser;
	}
	
	/**
	 * Prompt the user asking them where they would like to save the plot
	 * as an image. 
	 * @param plot The plot for which this dialog box will hover over.
	 * @return null if the user canceled, String [] {extension,fileName} 
	 * otherwise.
	 */
	private static String[] promptUser(Component plot){
		int option;
		
		//I don't like this hack but it is easier than pushing another
		//parameter onto the stack to tell us that plot is indeed a
		//braininfopane.
		if(plot instanceof BrainInfoPane){
			chooser.setDialogTitle("Save Brain Images As ...");
			option = chooser.showDialog(plot, "Save Brain Images");
		}else{
			chooser.setDialogTitle("Save Plot As ...");
			option = chooser.showDialog(plot, "Save Plot As ...");
		}
		
		if (option == JFileChooser.APPROVE_OPTION) {
			File file = chooser.getSelectedFile();
			String fileName = file.getAbsolutePath();
			
			// Sets the selected file extension based 
			//on the selected file filter.
			String extension = chooser.getFileFilter().getDescription();
			if (!fileName.endsWith("." + extension) ) {
				fileName += "." + extension;
			}
			return new String[] {extension,fileName};
		}
		return null;
	}

	/**
	 * Write out the image of the plot to the location specified by fileName.
	 * @param bi the image to save.
	 * @param extension the type of image {gif,jpeg,png,etc}
	 * @param fileName the location to save the image to.
	 */
	private static void writeImage(BufferedImage bi, String extension, 
			String fileName){
		
		try {
			ImageIO.write(bi, extension, new File(fileName));
		} 
		catch (IOException e) {
			GlobalVariablesFunctions.showErrorMessage(
					"Unable to save image.");
		}
	}
	
	/**
	 * Standard image saving function used by the majority of the plots. 
	 * This includes all of the abstract plots (excluding the abstract 
	 * functional plots). 
	 * @param chartPanel the chart panel holding the plot we want to write
	 * out as an image.
	 */
	public static void doSaveAs(ChartPanel chartPanel) {
		JFreeChart chart = chartPanel.getChart();
		if (chart != null) {
			String[] result = promptUser((Component) chartPanel);
			String extension,fileName;
			
			if(result != null){
				extension = result[0];
				fileName = result[1];
			}else{
				return;
			}
			
			BufferedImage image = chart.createBufferedImage(
					chartPanel.getWidth(),chartPanel.getHeight());
			writeImage(image, extension, fileName);

		} else {
			GlobalVariablesFunctions.showErrorMessage(
					"No plot has been loaded yet.");
		}
	}
	
	/**
	 * Save an AbstractFunctionalPlot as an image.
	 * @param AFPlot
	 * @param selectedIndex
	 * @param rowValue
	 * @param colValue
	 * @param mCharts
	 * @param mShowAverage
	 * @param mXAxisNames
	 * @param mAverages
	 */
	public static void saveAFPlot(AbstractFunctionalPlot AFPlot, 
								int selectedIndex, 
								int rowValue, 
								int colValue, 
								ChartPanel[][] mCharts,
								boolean mShowAverage,
								ArrayList<ArrayList<String>> mXAxisNames,
								ChartPanel[] mAverages){
		
		String[] result = promptUser((Component) AFPlot);
		String extension,fileName;
		
		if(result != null){
			extension = result[0];
			fileName = result[1];
		}else{
			return;
		}
		
		try {
			int group = selectedIndex;

			/* Retrieves the width and height of a single chart, 
			 * using one of the currently-displayed charts to do so. 
			 * This is used for drawing eachchart on the main image 
			 * to be saved. */
			ChartPanel cp = mCharts[rowValue][colValue];
			int chartWidth = cp.getWidth();
			int chartHeight = cp.getHeight();
			
			// Creates the overall image to be saved 
			// by concatenating each chart image that is created.
			BufferedImage bigImage;
			if (mShowAverage) {
				bigImage = new BufferedImage(chartWidth * mCharts[0].length 
											+ chartWidth,
											 chartHeight * mCharts.length, 
											 BufferedImage.TYPE_INT_RGB);
			} else {
				bigImage = new BufferedImage(chartWidth * mCharts[0].length,
						 chartHeight * mCharts.length, 
						 BufferedImage.TYPE_INT_RGB);
			}
			Graphics2D g = bigImage.createGraphics();
			int x = 0;
			int y = 0;
			
			for (int i = 0; i != mCharts.length; i++) {
				x = 0;
				for (int j = 0; j != mCharts[0].length; j++) {
					
					// Clones the chart such that any modifications made for
					// the image to be saved will not affect what is 
					// currently being displayed.
					JFreeChart chart = (JFreeChart) mCharts[i][j].getChart()
																 .clone();
					
					// Only displays the subject name for the first row.
					if (i == 0) {
						chart.setTitle(mXAxisNames.get(group).get(j));
					} else {
						chart.setTitle("");
					}
					
					// Only displays the range axis for the first column 
					// and only displays the domain axis for the last row.
					XYPlot plot = chart.getXYPlot();
					plot.getRangeAxis().setVisible(j == 0);
					plot.getDomainAxis().setVisible(i == mCharts.length - 1);
					
					// Adds the image of the current chart to the overall 
					// image to be saved.
					BufferedImage image = chart.createBufferedImage(
							chartWidth, chartHeight);
					g.drawImage(image, x, y, chartWidth, chartHeight, null);
					
					x += chartWidth;
				}
				
				if (mShowAverage) {
					JFreeChart chart = (JFreeChart) mAverages[i].getChart()
																.clone();
					
					if (i == 0) {
						chart.setTitle("Average");
					} else {
						chart.setTitle("");
					}
					
					XYPlot xyplot = chart.getXYPlot();
					xyplot.getRangeAxis().setLabel("");
					xyplot.getDomainAxis().setVisible(i == 
													mAverages.length - 1);
					
					BufferedImage image = chart.createBufferedImage(
												chartWidth, chartHeight);
					g.drawImage(image, x, y, chartWidth, chartHeight, null);
				}
				
				y += chartHeight;
			}
			ImageIO.write(bigImage, extension, new File(fileName));
		} catch (IOException e) {
			GlobalVariablesFunctions.showErrorMessage(
					"Unable to save the plots.");
		} catch(CloneNotSupportedException e){e.printStackTrace();}
	}
	
	/**
	 * Save a Brain and behavior plot as an image.
	 * @param bbplot
	 * @param group
	 * @param behav
	 * @param cond
	 * @param mCharts
	 */
	public static void saveBBScoresPlot(BrainBehavScoresPlot bbplot,
										int group, int behav, int cond,
										ArrayList<ArrayList
										<ArrayList<ChartPanel>>> mCharts){
				
		String[] result = promptUser((Component) bbplot);
		String extension,fileName;
		
		if(result != null){
			extension = result[0];
			fileName = result[1];
		}else{
			return;
		}
		
		// Retrieves the width and height of a single chart, using the
		// currently-displayed chart to do so. This is used for drawing each
		// chart on the main image to be saved.
		
		ChartPanel cp = mCharts.get(group).get(behav).get(cond);
		int chartWidth = cp.getWidth();
		int chartHeight = cp.getHeight();
		
		// Creates the overall image to be saved by concatenating each chart
		// image that is created.
		ArrayList<ArrayList<ChartPanel>> groupPanel = mCharts.get(group);
		ArrayList<ChartPanel> behavs = groupPanel.get(0);
		BufferedImage bigImage = new BufferedImage(
										chartWidth * groupPanel.size(),
										 chartHeight * behavs.size(), 
										 BufferedImage.TYPE_INT_RGB);
		Graphics2D g = bigImage.createGraphics();
		int x = 0;
		int y = 0;
	
		// Retrieves each individual chart image to be saved.
		for (int i = 0; i != groupPanel.size(); i++) {
			y = 0;
			behavs = groupPanel.get(i);
			for (int j = 0; j != behavs.size(); j++) {
				ChartPanel conditionPanel = behavs.get(j);
				JFreeChart chart = conditionPanel.getChart();
				
				BufferedImage image = chart.createBufferedImage(chartWidth, 
						chartHeight);
				g.drawImage(image, x, y, chartWidth, chartHeight, null);
				
				y += chartHeight;
			}
			x += chartWidth;
		}
		writeImage(bigImage,extension,fileName);
	}
	
	/**
	 * Save the loaded brain slices as an image file.
	 * @param bip the brain info pane which contains the brain slices.
	 */
	public static void saveBrainSlices(BrainInfoPane bip){
		String[] result = promptUser((Component) bip);
		String extension,fileName;
		
		if(result != null){
			extension = result[0];
			fileName = result[1];
		}else{
			return;
		}
		
		BufferedImage bi = bip.imageMontagePanel.concatenateImages();
		writeImage(bi, extension, fileName);
	}
	
	/**
	 * Save the scatter plot as an image file.
	 * @param scatterPlot the scatter plot to save.
	 */
	public static void saveESP(EnhancedScatterPlot scatterPlot){
		String[] result = promptUser((Component) scatterPlot);
		String extension,fileName;
		
		if(result != null){
			extension = result[0];
			fileName = result[1];
		}else{
			return;
		}
		
		int width = scatterPlot.getWidth();
		int height = scatterPlot.getHeight();
		
		BufferedImage capture = new BufferedImage(width,height,
												BufferedImage.TYPE_INT_RGB);
		scatterPlot.paint(capture.createGraphics());
		writeImage(capture,extension,fileName);
	}
}

