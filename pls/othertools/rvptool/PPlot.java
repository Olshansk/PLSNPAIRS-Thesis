package pls.othertools.rvptool;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;
import java.util.Set;
import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.filechooser.FileFilter;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.labels.StandardCategoryItemLabelGenerator;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.category.LineAndShapeRenderer;
import org.jfree.data.category.DefaultCategoryDataset;

/**
 * Class for generating the prediction vs pc # metaplot. Group data is passed in
 * and then the plot is generated.
 */
class PPlot implements ActionListener{

	final JFrame contentFrame; //main frame which components are drawn on.
	final JFreeChart chart; //the prediction by pc # plot.
	final DefaultCategoryDataset[] mmDatasets; //the datasets for the plot.

	//These fields should also only ever be assigned once.
	LineAndShapeRenderer render; //rendering object for the chart.
	PPlotData data; //loaded means/medians for the requested groups.
	JFileChooser fd; //a file choosing dialog when the user wants to save vals.
	
	private boolean mPlotPercentAccuracy = false; // if true, plot prediction percent accuracy instead of
    											  // posterior probability on Y axis

	public static void main(String args[]){
		ArrayList<CurveGroup> setupInfo = new ArrayList<CurveGroup>();

		//create three groups and test that session means/medians can be loaded
		//for session numbers #1,2,3 across result files pc#8,12
		setupInfo.add(new CurveGroup("G1",
			"/home/anita/plsnpairs/grady/results/y246_5cond_aug4_2010/" +
			"y246_5cond_aug4_2010_$pc_NPAIRSJresult.mat",1,"8,12"));
		setupInfo.add(new CurveGroup("G2",
			"/home/anita/plsnpairs/grady/results/y246_5cond_aug4_2010/" +
			"y246_5cond_aug4_2010_$pc_NPAIRSJresult.mat",2,"8,12"));
		setupInfo.add(new CurveGroup("G3",
			"/home/anita/plsnpairs/grady/results/y246_5cond_aug4_2010/" +
			"y246_5cond_aug4_2010_$pc_NPAIRSJresult.mat",3,"8,12"));
		// test percent accuracy
		String predType = "Percent Accuracy";
		new PPlot(setupInfo, predType);
	}

	PPlot(ArrayList<CurveGroup> setupInfo, String predType) {

		if (predType.equals("Percent Accuracy")) {
			mPlotPercentAccuracy = true;
		}
		
		/*Trim whitespace from result file paths*/
		trimWhitespace(setupInfo);

		/* Begin Laying out the widgets and drawing the plot */
		contentFrame = new JFrame();
		JPanel controlBar = new JPanel();
		JCheckBox labelEnable = new JCheckBox("Show Prediction Values");
		
		JRadioButton selectMeans = new JRadioButton("Means");
		JRadioButton selectMedians = new JRadioButton("Medians");
		ButtonGroup typeSelector = new ButtonGroup();

		JButton saveMeans = new JButton("Save means");
		JButton saveMedians = new JButton("Save medians");
		JButton saveBoth = new JButton("Save both");

		mmDatasets = generateDataset(setupInfo, mPlotPercentAccuracy);
		chart = makePlot();

		selectMeans.setSelected(true);
		typeSelector.add(selectMeans);
		typeSelector.add(selectMedians);

		labelEnable.setSelected(true);
		labelEnable.setActionCommand("labels");
		labelEnable.addActionListener(this);
		selectMeans.setActionCommand("meansButton");
		selectMeans.addActionListener(this);
		selectMedians.setActionCommand("mediansButton");
		selectMedians.addActionListener(this);
		saveMeans.setActionCommand("means");
		saveMeans.addActionListener(this);
		saveMedians.setActionCommand("medians");
		saveMedians.addActionListener(this);
		saveBoth.setActionCommand("means and medians");
		saveBoth.addActionListener(this);

		controlBar.add(labelEnable);
		controlBar.add(selectMeans);
		controlBar.add(selectMedians);
		controlBar.add(saveMeans);
		controlBar.add(saveMedians);
		controlBar.add(saveBoth);

		contentFrame.getContentPane().setLayout(new BorderLayout());
		contentFrame.getContentPane().add(new ChartPanel(chart));
		contentFrame.getContentPane().add(controlBar, BorderLayout.SOUTH);
		contentFrame.setTitle("Prediction VS PC Meta Plot");
		contentFrame.pack();
		contentFrame.setVisible(true);
		contentFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		/*End*/

	}

	private void trimWhitespace(ArrayList<CurveGroup> setupInfo){
		for(CurveGroup rg : setupInfo){
			rg.setFilename(rg.getFilename().trim());
		}
	}
	/**
	 * Generate the prediction by pc # plot.
	 * @return the chart containing the plotted groups.
	 */
	private JFreeChart makePlot(){
		
		CategoryPlot plot;
		DefaultCategoryDataset dataset = mmDatasets[0]; //means by default.
		ValueAxis valueAxis;
		JFreeChart chart;
		LineAndShapeRenderer renderer;
		double chartUpperBound; //add 10% height increase to the top of the plot
                               //do this so labels aren't cramped at the top.
		
		String rangeAxisLabel = "Prediction";
		if (mPlotPercentAccuracy) {
			rangeAxisLabel = "Prediction Percent Accuracy / 100";
		}
		chart = ChartFactory.createLineChart(
            "Prediction vs PC #",       // chart title
            "PC #",                    // domain axis label
            rangeAxisLabel,              // range axis label
            dataset,                   // data
            PlotOrientation.VERTICAL,  // orientation
            true,                      // include legend
            true,                      // tooltips
            false                      // urls
        );

		plot = chart.getCategoryPlot();
		valueAxis = plot.getRangeAxis();
		valueAxis.setAutoRange(true);
		chartUpperBound = valueAxis.getUpperBound() +
				         (valueAxis.getUpperBound() * .10);
		valueAxis.setUpperBound(chartUpperBound);
		
		renderer = (LineAndShapeRenderer) plot.getRenderer();
		renderer.setBaseShapesVisible(true);
		renderer.setBaseItemLabelGenerator(
				new StandardCategoryItemLabelGenerator());
		renderer.setBaseItemLabelsVisible(true);
		render = renderer;
		
		return chart;
	}

	/**
	 * Generate the dataset required by the CategoryPlot. A category plot needs
	 * a dataset so it can know which points it needs to plot.
	 * @param setupInfo The groups for which data will be loaded. This involves
	 * loading the specific result files, and calculating the means and medians.
	 * Each group has a given name and corresponds to a set of means/medians
	 * for a particular session/subject over a set of result files.
	 * @param plotPercentAccuracy If true, plot prediction percent accuracy
	 * instead of posterior probability.
	 * @return the dataset (the points to plot). index 0 contains the points
	 * for the means, index 1 the points for the medians.
	 */
	private DefaultCategoryDataset[] generateDataset(
			ArrayList<CurveGroup> setupInfo, boolean plotPercentAccuracy){

		data = new PPlotData(setupInfo, plotPercentAccuracy);
		DefaultCategoryDataset datasetMeans = new DefaultCategoryDataset();
		DefaultCategoryDataset datasetMedians = new DefaultCategoryDataset();

		Map<String,ArrayList<Double>> fileMedians = data.getFileMedians();
		Map<String,ArrayList<Double>> fileMeans = data.getFileMeans();
		
		String invalidGroups = "";
		for (CurveGroup group : setupInfo){

			for(String num : group.getParsedNumbers()){

				String file = group.getFilename().replace("$", num);

				if(!fileMeans.containsKey(file) ||
				   !fileMedians.containsKey(file)){
					continue; //this file was not loaded for whatever reason.
					          //so continue onto the next file.
							  //an error report is in data.getErrorString().
				}

				int session = group.getCurveUnit() - 1; //session # are 1-base.
				ArrayList<Double> mean = fileMeans.get(file);

				//file was loaded because some other curve can use it properly.
				if(group.getCurveUnit() > mean.size()){
					invalidGroups += "Curve (" + group.getLabel() 
					+ ") has specified " +	"an invalid subject (" 
					+ (session+1) + ")\n";
					break; //don't try another of the other files since 
						//they wont contain the subject either
				}
				
				
				ArrayList<Double> median = 	fileMedians.get(file);
				
				int pcNum = Integer.parseInt(num);
				double sessionMean = mean.get(session);
				double sessionMedian = median.get(session);
				String label = group.getLabel();
				String pcId = Integer.toString(pcNum);

				if (sessionMean != -1){ //-1 means session has no mean.
					datasetMeans.addValue(sessionMean, label, pcId);
				}

				if (sessionMedian != -1){ //-1 means session has no median.
					datasetMedians.addValue(sessionMedian, label, pcId);
				}
			}
		}

		//get culminated errors encountered when reading in mat files.
		String error = null;
		String loadingError = data.getErrorString();
		if(!loadingError.equals("") && !invalidGroups.equals("")){
			error = invalidGroups + "\n" + loadingError;
		}
		else if(loadingError.equals("") && !invalidGroups.equals("")){
			error = invalidGroups;
		}
		else if(!loadingError.equals("") && invalidGroups.equals("")){
			error = loadingError;
		}
		
		if(error != null){
		JOptionPane.showMessageDialog(contentFrame, error +
					"\n The Plot will be lacking information.",
					"Error", JOptionPane.ERROR_MESSAGE);
		}

		return new DefaultCategoryDataset[] {datasetMeans,datasetMedians};
	}

	/**
	 * Opens A save dialog asking the user where they want to save the
	 * means and/or medians.
	 * @param command A string denoting the type of data the user wants to save.
	 * Means, medians, or both.
	 */
	private void openSaveDialog(String command){

		if(fd == null){
			fd = new JFileChooser(".");

			fd.addChoosableFileFilter(new FileFilter() {

				@Override
				public boolean accept(File f) {
					return f.getName().endsWith(".txt") || f.isDirectory();
				}

				@Override
				public String getDescription() {
					return "txt";
				}
			});
			fd.setPreferredSize(new Dimension(680, 480));
			fd.setMultiSelectionEnabled(false);
		}
		
		fd.setDialogTitle("Select location to save " + command);
		
		int option = fd.showSaveDialog(contentFrame);

		if(option == JFileChooser.APPROVE_OPTION){
			File location = fd.getSelectedFile();
			JOptionPane question = new JOptionPane();
			String message;

			if(fd.getFileFilter().getDescription().equals("txt")){
				if(!location.getName().endsWith(".txt")){
					location = new File(location.getAbsolutePath()+".txt");
				}
			}

			if(location.exists()){
				message = "File already exists. Overwrite?";
				int retval = question.showConfirmDialog(contentFrame,message,
					"Overwrite file?",JOptionPane.YES_NO_OPTION,
					JOptionPane.QUESTION_MESSAGE);

				if(retval != JOptionPane.OK_OPTION) return;

				if(!location.canWrite()){
				message = "Cannot write to specified location. "+
						"Check permissions.";
				question.showMessageDialog(contentFrame, message,
				"Could not write to location", JOptionPane.INFORMATION_MESSAGE);
				return;
				}
			}

			writeVals(command,location);
		}
	}

	private boolean abortSaving(){
		ArrayList<String> pcnums = null;
		
		for(CurveGroup curve : data.getCurves()){
			
			//Check that pc nums are the same for all curves.
			if(pcnums == null) pcnums = curve.getParsedNumbers();
			else if(!pcnums.equals(curve.getParsedNumbers())){
				JOptionPane.showMessageDialog(contentFrame, 
						"Could not save values. Must include the same" +
						"pcs for all curves.","Error",
						JOptionPane.ERROR_MESSAGE);
				return false;
			}
			
			//Check that all pc files for this curve have data.
			String filename;
			for(String pc : pcnums){
				filename = curve.getFilename().replace("$", pc);
				if(data.getFileMeans().get(filename) == null){
					JOptionPane.showMessageDialog(contentFrame,
							"Could not save values. All pcs must be the" +
							"\nsame for all curves and each pc must contain" +
							" a value.", "Error", JOptionPane.ERROR_MESSAGE);
					return false;
				}
			}
		}
		return true;
	}
	
	/**
	 * Writes the means and/or medians out to the given location.
	 * @param command A string denoting the type of data the user wants to save.
	 * @param location The location to save the means/medians to.
	 */
	private void writeVals(String command, File location){
		if(!abortSaving()) return;
		
		Map<String,ArrayList<Double>> means = data.getFileMeans();
		Map<String,ArrayList<Double>> medians = data.getFileMedians();

		try{
			FileOutputStream fWriter = new FileOutputStream(location);
			OutputStreamWriter out;

			/*The java documentation has this to say about the ascii and utf-8
			charsets. "Every implementation of the Java platform is required
			 to support the following standard charsets."*/

			Charset cs;
			if(Charset.isSupported("UTF-8")) cs = Charset.forName("UTF-8");
			else if (Charset.isSupported("US-ASCII"))
				cs = Charset.forName("US-ASCII");
			else cs = Charset.defaultCharset();

			out = new OutputStreamWriter(fWriter, cs);

			String header = null;
			String meansRow = null;
			String medianRow = null;
			
			int unit;
			boolean badcurve = false;
			boolean wroteHeader = false;
			
			for(CurveGroup curve : data.getCurves()){
				unit = curve.getCurveUnit() - 1; //unit are 1 based.
				
				//All curves should have the same number of pcs 
				//see: abortSaving()
				if(header == null){
					header = ",,";
					ArrayList<String> pcs = curve.getParsedNumbers();
					int i = 0;
					for(; i < pcs.size() - 1; i++)
						header += Integer.parseInt(pcs.get(i)) + " pc,";
					header += Integer.parseInt(pcs.get(i)) + " pc";  
				}
				
				meansRow = null;
				medianRow = null;
				badcurve = false;
				
				for(String pc : curve.getParsedNumbers()){
					String filename = curve.getFilename().replace("$", pc);
										
					//invalid session/run/split_obj specified by this group.
					//this means this curve group is invalid.
					if(unit > means.get(filename).size() - 1){
						meansRow = null;
						medianRow = null;
						break;
					}
					
					if(meansRow == null){
						meansRow = data.getSplitObjType().get(filename);
						meansRow += " #" + curve.getCurveUnit();
						meansRow += "," + curve.getLabel();
						medianRow = meansRow;
					}
					
					Double meanVal = means.get(filename).get(unit);
					Double medianVal = medians.get(filename).get(unit);
					
					if(meanVal == -1){meansRow += "," + "NA";}
					else{meansRow += "," + means.get(filename).get(unit);}
					
					if(medianVal == -1){medianRow += "," + "NA";}
					else{medianRow += "," + medians.get(filename).get(unit);}
					
				}
								
				//Does this curve have values that were plotted? If not pass on
				//printing it out.
				if((command.equals("means") && meansRow == null) ||
				(command.equals("medians") && medianRow == null)){
					badcurve = true;
				}
				if(meansRow == null && medianRow == null){
					badcurve = true;
				}
				
				if(!badcurve){
					if(!wroteHeader){
						out.write(header + "\n");
						wroteHeader = true;
					}
					if(command.equals("means")){
						out.write(meansRow + "\n");
					}else if(command.equals("medians")){
						out.write(medianRow + "\n");
					}else{
						out.write(meansRow + "\n");
						out.write(medianRow + "\n");
					}
				}
			}
			out.close();
		}catch(FileNotFoundException e){
			JOptionPane.showMessageDialog(contentFrame, "Cannot Write to file."
					+" Saving the file has failed. " +
					"Possibly a permissions error.");
		}catch(IOException e){
			JOptionPane.showMessageDialog(contentFrame, "An IOException" +
					"has occured. This is a bad sign and may imply a critical" +
					"system error. \n" + e.getMessage());
		}
	}

	public void actionPerformed(ActionEvent e){
		if (e.getActionCommand().equals("labels")) {
			boolean val = render.getBaseItemLabelsVisible();
			render.setBaseItemLabelsVisible(!val);

			/* Extend the categoryItemLabelGenerator class to create different
			labels and then use it by passing it to
			render.setBaseItemLabelGenerator()*/

		}
		else if(e.getActionCommand().equals("meansButton")){
			((CategoryPlot) chart.getPlot()).setDataset(mmDatasets[0]);
		}
		else if(e.getActionCommand().equals("mediansButton")){
			((CategoryPlot) chart.getPlot()).setDataset(mmDatasets[1]);
		}
		else{
			openSaveDialog(e.getActionCommand());
		}
	}
}
