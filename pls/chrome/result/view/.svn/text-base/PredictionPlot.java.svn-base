package pls.chrome.result.view;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Scanner;
import java.util.TreeSet;

import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTabbedPane;

import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.CategoryAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.labels.BoxAndWhiskerToolTipGenerator;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.renderer.category.AbstractCategoryItemRenderer;
import org.jfree.data.Range;
import org.jfree.data.statistics.DefaultBoxAndWhiskerCategoryDataset;

import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.NPairsResultModel;
import pls.chrome.result.model.ResultModel;
import pls.shared.GlobalVariablesFunctions;
import pls.shared.PredictionPlotFilter;
import extern.NpairsBoxAndWhiskerRenderer;
import extern.NpairsBoxWhiskerAxisFix;
import extern.PlotBySplitClassRenderer;



/**
 * Note: This prediction plot does not yet support full plot persistence.
 * Enabling this persistence is a trivial matter and can be done by uncommenting
 * the code in createPlot() and adding an updateSettings() method nearly
 * identical to DesignLatentVariablesPlot's updateSettings(). This function
 * removes persistence information when the file is unloaded from the result
 * viewer. Currently tab selected persistence works and persists for a given
 * file even when it is unloaded from the result viewer.
 * 
 * Note: There is the assumption that the subj_label array lists the subject 
 * numbers consecutively. That is {subj1,subj1,subj1,subj2,subj3,subj3...}
 * and not {subj1,subj2,subj1,subj3,subj2,subj1...}
 */

public class PredictionPlot extends AbstractPlot{

	private static final long serialVersionUID = -4379135469294012336L;

	public static final int ENTER_KEY = 10;
	
	// The regular expressions used to for parsing the input fields related
	// to CV dim value selection.
	public static final String NUMBER_REGEX = "\\d+";
	public static final String HYPHEN_RANGE_REGEX = "\\d+\\s*-\\s*\\d+";
	public static final String COMMA_RANGE_REGEX = "(\\d+\\s*,\\s*)+\\d+";
	public static final String MULTI_RANGE_REGEX = "(\\d+|\\d+\\s*-\\s*\\d+)(\\s*,\\s*(\\d+|\\d+\\s*-\\s*\\d+))+";
	
	private static final String MEANVAL = "MEAN";
	private static final String MEDIANVAL = "MEDIAN";
	private static final String MANDMVAL = "MEANMEDIAN";
	
	private double[][] mPPTrue;
	
	private double[][] mSplitVols = null;
	private double[] mSplitObjLabels = null;
	private double[] mClassLabels = null;
	private List<String> mClassNames;
	private String mSplitType;
	
	private JButton saveMeanButton;
	private JButton saveMedianButton;
	private JButton saveBothButton;
	private JButton regressionButton;

	private JPanel ButtonControl;
	
	private String regressionFileName;

	private JFileChooser chooser;
	private JFileChooser regressionChooser;
	private PredictionPlotFilter filter;

	private JTabbedPane mTabs;
	private ChartPanel[] mPredChartPanel;

	private ArrayList<ArrayList<Double>> mSplitObjValues = null;
	private ArrayList<Double> regressionVals;
	private double[][][][] mSplitByClassVals = null;
	
	private RegressionPlot regressionPlot;
	private ChartPersistence chartPersistence;
	
	public PredictionPlot(String title, GeneralRepository repository,
			RegressionPlot regressionPlot) {
		
		super(title, repository, GlobalVariablesFunctions.NPAIRS);

		this.regressionPlot = regressionPlot;
		
		//if we're running pls
		if (mResultFilePaths.isEmpty()) {
			ArrayList<AttachDetachOption> mPlots = mManager.getPlots();
			ArrayList<Boolean> mAttachedPlots = mManager.getAttachedPlots();
			
			for (int i = 0; i < mPlots.size(); i++) {
				if (mPlots.get(i) == this) {
					mPlots.remove(i);
					mAttachedPlots.remove(i);
				}
			}
		}
		//testMeanMedianCalculation();
	}
		
	private String getSplitType(){
		int fileIndex = mFileComboBox.getSelectedIndex();
		String path = mResultFilePaths.get(fileIndex);
		NPairsResultModel model = mRepository.getNpairsModel(path);
		return model.getSplitType();
	}
	
	@Override
	public void actionPerformed(ActionEvent e) {
		if(e.getSource() == saveMeanButton) openSaveDialog(MEANVAL);
		else if(e.getSource() == saveMedianButton) openSaveDialog(MEDIANVAL);
		else if(e.getSource() == saveBothButton) openSaveDialog(MANDMVAL);
		else if(e.getSource() == regressionButton){
			if(loadRegressionDialog()){
				regressionPlot.drawPlot(
						regressionVals,
						getSplitObjMeans(),
						getSplitObjMedians(),
						getCurrentFile(),
						regressionFileName,
						getSplitType());
			}
		}
		else{
			int fileIndex = mFileComboBox.getSelectedIndex();
			NPairsResultModel model = mRepository.getNpairsModel(
					mResultFilePaths.get(fileIndex) );
						
			if(!checkValidSplitObjLabels(model.getSplitObjLabels())){
				JOptionPane.showMessageDialog(this, 
				"The split object labels for this result file are either " +
				"not monotonic\n" +	"or at least one of the labels has a " +
				"value that is greater than the\n" +
				"the number of labels. The plot will not be plotted.", 
				"Bad split object labels", JOptionPane.ERROR_MESSAGE);
			}
			
			super.actionPerformed(e);
		}
	}
	
	public double[][] getRangeData(ResultModel model) {
		return ((NPairsResultModel) model).getPPTrueClass();
	}
	
	
	@Override
	public void makeChart(int fileIndex) {
		if (mPredChartPanel == null) {
			mPredChartPanel = new ChartPanel[3];
		}

		if(chartPersistence == null){
			chartPersistence = new ChartPersistence();
		}

		initChartPersistence(getCurrentFile());
		
		NPairsResultModel model = mRepository.getNpairsModel(
				mResultFilePaths.get(fileIndex) );
		mPPTrue = model.getPPTrueClass();
		mSplitObjLabels = model.getSplitObjLabels();
		mSplitVols = model.getSplitVols();
		mSplitType = model.getSplitType();
		mClassLabels = model.getClassLabels();
		mClassNames = model.getClassNames();
		
		if(mSplitType == null){
			mSplitType = "Split object";
		}
		//testSetup(mFileComboBox.getSelectedIndex());
		fillPanels(fileIndex);

		mTabs.setSelectedIndex(chartPersistence.getSelectedTab(
				getCurrentFile()));
		tabClickAction();
		
	}

	/**
	 * Save the means for each split object for the currently selected result file
	 * @param vals ArrayList of ArrayLists, each of which contain the means
	 * for a single split object.
	 */
	public void setSplitObjectValues(ArrayList<ArrayList<Double>> vals){
		mSplitObjValues = vals;
	}

	/**
	 * Get the saved split object values (means for each split object).
	 * @return  Arraylist of Arraylists each containing the means for a single
	 * split object
	 */
	public ArrayList<ArrayList<Double>> getSplitObjValues(){
		return mSplitObjValues;
	}

	@Override
	public void doSaveAs(){
		ChartPanel chartpanel = (ChartPanel) mTabs.getSelectedComponent();
		doSaveAs(chartpanel);
	}
	/**
	 * Debug function for dumping the values of the loaded regression data.
	 */
	private void dumpRegressionData(){
		for(double val : regressionVals)
			System.out.println(val);
	}
	
	/**
	 * Create a pop up for loading regression data.
	 * @return true if data was loaded properly, false otherwise.
	 */
	private boolean loadRegressionDialog(){
		if(regressionChooser == null){
			regressionChooser = new JFileChooser(".");
			if(filter == null){
				filter = new PredictionPlotFilter();
			}
			regressionChooser.addChoosableFileFilter(filter);
			regressionChooser.setPreferredSize(new Dimension(680,480));
			regressionChooser.setMultiSelectionEnabled(false);
		}

		int option = regressionChooser.showOpenDialog(this);
		if(option == JFileChooser.APPROVE_OPTION){
			File openLocation = regressionChooser.getSelectedFile();
			if(!openLocation.canRead()){
				JOptionPane.showMessageDialog(this, "File does not exist or" +
						"file cannot be read");
				return false;
			}
			if(loadRegressionData(openLocation)){
				regressionFileName = openLocation.getName();
				return true;
			}
		}
		return false; //failure to load data.
	}

	/**
	 * Load the regression data at the specified location.
	 * @param openLocation the file for which contains the regression data.
	 * @return true if data loaded correctly, false otherwise.
	 */
	private boolean loadRegressionData(File openLocation){
		ArrayList<Double> rVals = new ArrayList<Double>();

		try{
			Scanner scanner = new Scanner(openLocation);
			while(scanner.hasNextLine()){
				if(parseRegressionDataLine(scanner.nextLine(),rVals)){
					return false; //there was an error, return.
				}
			}
			regressionVals = rVals;
			return true;
		}catch(java.io.FileNotFoundException e){
			JOptionPane.showMessageDialog(this,"The selected file for loading" +
					"could not be found.");
			return false;
		}
	}

	/**
	 * Display an error message.
	 * @param message the warning message to display.
	 */
	private void displayReadError(String message){
		JOptionPane.showMessageDialog(this, message);
	}

	/**
	 * Parse a single line of regression data.
	 * @param line the line of regression data to be parsed.
	 * @param rvals a container holding any previously parsed data in which
	 * data parsed from this line will be added to. 
	 * @return true if there was a parse error, false otherwise.
	 */
	private boolean parseRegressionDataLine(String line, ArrayList<Double> rvals){
		try{
			//comment line, ignore.
			if(line.startsWith("#")) return false;
			else if(line.trim().equals("")) return false; //empty line.
			else if(line.contains(",")){
				String [] tokens= line.split(",");
				
				if(tokens.length != 3){
					String message = "Line: " + line + "\n" + 
					"Must be of the form: [SplitType #n,mean|median,val] " +
					"when using CSVs";
					displayReadError(message);
					return true;
				}
				//Value not include for this split, note this with a null.
				if(tokens[2].trim().toUpperCase().equals("NA")){
					rvals.add(null);
				}else{
					rvals.add(Double.parseDouble(tokens[2]));
				}
			}else{
				if(line.trim().toUpperCase().equals("NA")){
					rvals.add(null);
				}else{
					rvals.add(Double.parseDouble(line));
				}
			}
		}catch(NumberFormatException e){
			String message = "Line: " + line + "\n" + 
			"contained an unparsable value";
			displayReadError(message);
			return true;
		}
		return false;
	}

	/**
	 * Open a save dialog to save either the means, medians, or both of the
	 * currently selected result file.
	 * @param type indicator of whether the user wants to save medians, medians
	 * or both. type is either MEANVAL, MEDIANVAL or MANDMVAL.
	 */
	private void openSaveDialog(String type){
		
		//If the new plot by class and subject is selected then
		//the split by class vals should not be empty.
		boolean writeByClass = false;
		if(chartPersistence.getSelectedTab(getCurrentFile()) == 2){
			if(getSplitByClassVals() == null) return;
			writeByClass = true;
		}else{
			if(getSplitObjValues() == null) return;
		}
		
		if(chooser == null){
			chooser = new JFileChooser(".");
			if(filter == null){
				filter = new PredictionPlotFilter();
			}
			chooser.addChoosableFileFilter(filter);
			chooser.setPreferredSize(new Dimension(680, 480));
			chooser.setMultiSelectionEnabled(false);
		}
		
		int option = chooser.showSaveDialog(this);
		if (option == JFileChooser.APPROVE_OPTION) {
			File saveLocation = chooser.getSelectedFile();

			//Always append the extension of the set filter to the path.
			if(chooser.getFileFilter() == filter){
				if(!saveLocation.getName().endsWith(filter.EXTENSION)){
					saveLocation = new File(saveLocation.getAbsolutePath()
							+filter.EXTENSION);
				}
			}

			if(saveLocation.exists()){
				JOptionPane warning = new JOptionPane();
				String message = "Selected file already exists. " +
						"Are you sure you want to overwrite this "+
						"file?";

				int retval = warning.showConfirmDialog(this,message,
					"Overwrite file?",JOptionPane.YES_NO_OPTION,
					JOptionPane.QUESTION_MESSAGE);

				//Get the selected value, if null or not integer return.
				//Also return if OK was not selected.
				if(retval != JOptionPane.OK_OPTION){
					return;
				}
				
				//User cannot write to selected location.
				if(!canWrite(saveLocation)) return;
			}
			
			if(!writeByClass){
				writeMeanOrMedian(saveLocation,type);
			}else{
				try{
					writeMMClass(saveLocation,type);
				}catch(IOException e){
					JOptionPane.showMessageDialog(this, 
					"Could not write to file. Check your permissions.",
					"Error", JOptionPane.ERROR_MESSAGE);
				}
			}
		}
	}

	/**
	 * Determine whether the selected file is writable.
	 * @param saveLocation the file to be written to.
	 * @return true if the file is writable. False otherwise.
	 */
	private boolean canWrite(File saveLocation){
		if(!saveLocation.canWrite()){
			JOptionPane.showMessageDialog(this, "Cannot Write to file." +
					" Your file permissions may be incorrect.",
					"Could not write to location",
					JOptionPane.INFORMATION_MESSAGE);
			return false;
		}
		return true;
	}

	/**
	 * Write out the medians and medians for each subject and 
	 * its specific class.
	 * @param saveLocation The location to save the medians/medians
	 * @param type MEANVAL/MEDIANVAL/OTHER the type of value to write out.
	 * @throws IOException If writing the file failed.
	 */
	private void writeMMClass(File saveLocation, String type) throws IOException{
		String splitType = getSplitType();
		if(splitType == null) splitType = "Split Object";
		
		double[][][][] sbc = getSplitByClassVals();
		int rows = sbc.length;
		int subj = sbc[0].length;
		int cls  = sbc[0][0].length;
		int	vlength;
		double rmean = -1;
		double rmedian = -1;
		
		//We need to map each unique label to an array index.
		//Once mapped to an array index, we can figure out which subject NUMBER
		//belongs to which array index. Once we know the NUMBER we can find the
		//actual name by looking indexing that number into the list of names.
		int[] slMap =  getMapForLabels(mSplitObjLabels);
		int[] clMap = getMapForLabels(mClassLabels);
		
		List<Double> means= new ArrayList<Double>(30);
		String rmeanout;		
		String rmedianout;
				
		OutputStream fw = new FileOutputStream(saveLocation);
		/*The java documentation has this to say about the ascii and utf-8
		charsets. "Every implementation of the Java platform is required
		 to support the following standard charsets."*/

		Charset cs;
		if(Charset.isSupported("UTF-8")) cs = Charset.forName("UTF-8");
		else if (Charset.isSupported("US-ASCII"))
			cs = Charset.forName("US-ASCII");
		else cs = Charset.defaultCharset();
		
		Writer osw = new OutputStreamWriter(fw, cs);
		Writer bfw = new BufferedWriter(osw);
		List<Double> out = new ArrayList<Double>();
				
		for(int s = 0; s < subj; s++){
			for(int c = 0; c < cls; c++){
				
				if(type.equals(MEANVAL)){
					//mClassLabels are one relative
					bfw.write("\nsubject:" + slMap[s] 
							  + ",class:"  + mClassNames.get(c) 
							  +",mean:");
				
				}else if(type.equals(MEDIANVAL)){
					
					bfw.write("\nsubject:" + slMap[s] 
							  + ",class:" + mClassNames.get(c) 
							  +",median:");
				}else{
					bfw.write("\nsubject:" + slMap[s] 
					          + ",class:" + mClassNames.get(c));
					bfw.write("\nmean,median\n");
				}
				
				means.clear();
				for(int r = 0; r < rows; r++){
					rmean = 0;
					//no values for this (subject,class) for this row
					if(sbc[r][s][c] == null) continue;
					for(double v : sbc[r][s][c]){
						rmean += v;
					}
					rmean /= sbc[r][s][c].length;
					means.add(rmean);
				}
				
				vlength = means.size();
				rmeanout = "NA";
				rmedianout = "NA";
				if(vlength > 0){
					rmean = 0;
					for(double m : means){
						rmean += m;
					}
					rmean /= vlength;
					
					
					if(vlength % 2 == 0){
						rmedian = means.get(vlength / 2) 
						        + means.get(vlength / 2 - 1);
						rmedian /= 2;
					}else{
						rmedian = means.get(vlength /2);
					}
					rmeanout = String.valueOf(rmean);
					rmedianout = String.valueOf(rmedian);
				}
				
				//write out values
				
				if(type.equals(MEANVAL)){
					bfw.write(rmeanout);
				}else if(type.equals(MEDIANVAL)){
					bfw.write(rmedianout);
				}else{
					bfw.write(rmeanout+","+rmedianout+"\n");
				}
			}
		}
		bfw.close();
	}
	/**
	 * Write the means, medians, or both to a file. A written value of -1.0
	 * means that the split object has no mean/median.
	 * @param saveLocation the location to save to.
	 * @param type indicator of whether the user wants to save medians, medians
	 * or both. type is either MEANVAL, MEDIANVAL or MANDMVAL.
	 */
	private void writeMeanOrMedian(File saveLocation, String type){
				
		
		String splitType = getSplitType();
		if (splitType == null) splitType = "Split Object";
		try{
			FileOutputStream fWriter = new FileOutputStream(saveLocation);

			/*The java documentation has this to say about the ascii and utf-8
			charsets. "Every implementation of the Java platform is required
			 to support the following standard charsets."*/

			Charset cs;
			if(Charset.isSupported("UTF-8")) cs = Charset.forName("UTF-8");
			else if (Charset.isSupported("US-ASCII"))
				cs = Charset.forName("US-ASCII");
			else cs = Charset.defaultCharset();

			OutputStreamWriter out = new OutputStreamWriter(fWriter, cs);
			out.write('#' + getCurrentFile()+'\n');
			
			ArrayList<Double> means = getSplitObjMeans();
			ArrayList<Double> medians = getSplitObjMedians();
			
			if(type.equals(MEANVAL)){
				
				int splitObjN = 1;
				
				for (Double val : means) {
					out.write(splitType + " #" + splitObjN++ + ",");
					out.write("mean,");
					
					if(val == -1.0){
						out.write("NA");
					}else{
						out.write(val.toString());
					}
					out.write("\n");
				}
			}else if(type.equals(MEDIANVAL)){
				int splitObjN = 1;
				
				for (Double val : medians) {
					out.write(splitType + " #" + splitObjN++ + ",");
					out.write("median,");
					
					if(val == -1.0){
						out.write("NA");
					}else{
						out.write(val.toString());
					}
					out.write("\n");
				}
			}
			else{
				int splitObjN = 1;
				for(Double val : means){
					Double medianVal = medians.get(splitObjN -1);
					out.write(splitType + " #" + splitObjN++ + ",");
					out.write("mean,");
					if(val == -1.0){
						out.write("NA");
					}else{
						out.write(val.toString());
					}
					out.write(",median," + medianVal + "\n");
				}
			}
			
			out.close();
			fWriter.close();
		}catch(java.io.FileNotFoundException e){
			JOptionPane.showMessageDialog(this, "Cannot write to file." +
					" Saving the file has failed. " +
					"Possibly a permissions error.");
		}
		catch(IOException e){
			JOptionPane.showMessageDialog(this, "A critical IO." +
					" error has occurred." + "\n" + e);
		}
  	}

	/**
	 * Concatenate the values of two doubles together and return this as a 
	 * string. If either of the values is -1.0 return substitute NA for that
	 * value.
	 * @param v1 first number.
	 * @param v2 second number.
	 * @return concatenation of the two numbers with a space in between.
	 */
	private String valueString(double v1, double v2){
		if(v1 == -1.0 && v2 == -1.0){
			return "NA NA";
		}else if(v1 == -1.0){
			return "NA " + v2;
		}else if(v2 == -1.0){
			return v1 + " NA";
		}else{
			return v1 + " " + v2;
		}
	}
	
	/**
	 * Simple sanity test.
	 * @param fileIndex index of model we run the sanity test on.
	 */
	private void testSetup(int fileIndex){
		NPairsResultModel model = mRepository.getNpairsModel(
				mResultFilePaths.get(fileIndex) );

		//1.a session has no means whatsoever.
		//2.different sessions have different conditions.
		//i.e session1 has condition 1 and 2 and session 2 has condition
		//3 and 4.
		double[][] newPPTrue = {{9,3,2,2},
			                    {2,3,4,1},
			                    {9,3,2,2},
								{2,3,4,1}};

		double[][] newSplitVols = {{1,1,2,2},
		                           {3,3,2,2},
								   {4,4,1,1},
								   {2,2,4,4}};

		double[] newSplitObjLabels = {1,1,2,2};
		double[] newClassLabels    = {1,2,1,2};
		ArrayList<String> newClassNames = new ArrayList<String>();
		newClassNames.add("CA2");
		newClassNames.add("CA1");
		
		model.setPPTrueClass(newPPTrue);
		model.setSplitVols(newSplitVols);
		model.setSplitObjLabels(newSplitObjLabels);
		model.setClassLabels(newClassLabels);
		model.setClassNames(newClassNames);
		mPPTrue = model.getPPTrueClass();
		mSplitObjLabels = model.getSplitObjLabels();
		mSplitVols = model.getSplitVols();
		mClassLabels = model.getClassLabels();
	}

	/**
	 * Test method for checking whether the median/mean calculations
	 * are being carried out correctly.
	 */
	private void testMeanMedianCalculation(){
	  boolean error = false;
	  ArrayList<ArrayList<Double>> vals = new ArrayList<ArrayList<Double>>(4);
	  ArrayList<Double> splitObj1 = new ArrayList<Double>();
	  ArrayList<Double> splitObj2 = new ArrayList<Double>();
	  ArrayList<Double> splitObj3 = new ArrayList<Double>();
	  ArrayList<Double> splitObj4 = new ArrayList<Double>();
	  double mean1,median1;
	  double mean2,median2;
	  double mean3,median3;
	  double mean4,median4;

	  splitObj1.add(4.0); splitObj1.add(6.4); splitObj1.add(14.2);
	  mean1 = (4.0 + 6.4 + 14.2) / 3;
	  median1 = 6.4;

	  splitObj2.add(3.0);
	  mean2 = 3.0;
	  median2 = 3.0;

	  splitObj3.add(16.3); splitObj3.add(17.3); splitObj3.add(7.4); splitObj3.add(12.66);
	  mean3 = (16.3 + 17.3 + 7.4 + 12.66) / 4;
	  median3 = (12.66 + 16.3) /2;

	  splitObj4.clear();
	  mean4 = -1;
	  median4 = -1;

	  vals.set(0, splitObj1);
	  vals.set(1, splitObj2);
	  vals.set(2, splitObj3);
	  vals.set(3, splitObj4);
	  
	  setSplitObjectValues(vals);
	  ArrayList<Double> splitObjMeans = getSplitObjMeans();
	  ArrayList<Double> splitObjMedian = getSplitObjMedians();
	  if(splitObjMeans.get(0) != mean1){error = true;}
	  if(splitObjMeans.get(1) != mean2){error = true;}
	  if(splitObjMeans.get(2) != mean3){error = true;}
	  if(splitObjMeans.get(3) != mean4){error = true;}
	  if(splitObjMedian.get(0) != median1){error = true;}
	  if(splitObjMedian.get(1) != median2){error = true;}
	  if(splitObjMedian.get(2) != median3){error = true;}
	  if(splitObjMedian.get(3) != median4){error = true;}

	  if(error) System.out.println("MEDIAN MEAN CALCULATIONS ARE INCORRECT!");

	}
	/**
	 * Testing function. Dumps the PP array.
	 */
	private void dumpPP(){
		for (int i = 0; i < mPPTrue.length; i++) {
			for (int j = 0; j < mPPTrue[0].length; j++) {
				System.out.print(mPPTrue[i][j] + ",");
			}
			System.out.println("\n");
		}
	}

	/**
	 * Testing function. Dumps the mapping made from splitVols and
	 * subjLabels.
	 */
	private void dumpMap(){
		double[][] map = transform();
		for(int i = 0; i < map.length; i++){
			for(int j = 0; j < map[0].length; j++){
				System.out.print(map[i][j] + ",");
			}
			System.out.println();
		}
	}

	/**
	 * Creates the mapping from splitVols and splitObjLabels.
	 * @return the mapping, the same dimensions as splitVols.
	 */
	private double[][] transform(){
		int rows = mSplitVols.length;
		int columns = mSplitVols[0].length;

		int vol = -1;
		double[][] map = new double[rows][columns];

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				vol = (int) mSplitVols[i][j];
				if (vol > 0) {
					map[i][j] = mSplitObjLabels[vol-1];
				}else{
					map[i][j] = -1;
				}
			}
		}
		return map;
	}

	/**
	 * Creates the dataset for the prediction plot (by split object). This is another
	 * method for calculating the same dataset as getBySplitObjDataset() and is
	 * not used. This method does not have the same precondition as below.
	 * @return The dataset for the plot (the means of each split object for each row)
	 */
	private DefaultBoxAndWhiskerCategoryDataset getSplitObjDataset(){
		DefaultBoxAndWhiskerCategoryDataset classDataset = new
				DefaultBoxAndWhiskerCategoryDataset();

		int numSplitObj = getNumSplitObjects();
		double[][] map = transform();
		
		for(int splitObj = 1; splitObj <= numSplitObj; splitObj++){
			ArrayList<Double> splitObjMean = new ArrayList<Double>();
			double beforeDiv;
			int count;
			for(int i = 0; i < mSplitVols.length; i++){ //row
				beforeDiv = 0;
				count = 0;

				for(int j = 0; j < mSplitVols[0].length; j++){ // column
					if(map[i][j] == splitObj){
						if(mPPTrue[i][j] != -1){
							beforeDiv += mPPTrue[i][j];
							count++;
						}
					}
				}
				if (count > 0) {
					splitObjMean.add((beforeDiv / (double) count));
				}
			}

			classDataset.add(splitObjMean, 0, splitObj);
		}
		return classDataset;
	}

	/**
	 * Given a label array (Either subject or class labels)
	 * reduce this array such that we can map a particular subject/class
	 * to a unique array index.
	 * @param labels the labels to create a mapping for.
	 * @return an array that contains each unique label without duplicates in
	 * ascending order.
	 */
	private int[] getMapForLabels(double[] labels){
		TreeSet<Integer> sorter = new TreeSet<Integer>();
		int[] map;
		for(double label : labels)
			sorter.add((int) label);
		
		
		map = new int[sorter.size()];
		
		int i = 0;
		for(int n : sorter) map[i++] = n;
		return map;
	}
	/**
	 * A terribly ugly function that returns a 4d array... 
	 * @param r The number of rows.
	 * @param sn The number of subjects. 
	 * @param cn The number of classes.
	 * @return Create an r X sn X cn x N (where N can be any size) arraylist.
	 */
	private List<List<List<List<Double>>>> blankArray(int r,int sn, int cn){
		List<List<List<List<Double>>>> rows = new ArrayList<List<List<List<Double>>>>(r);
		List<List<List<Double>>> subjects;
		List<List<Double>> classes;
		
		for(int i = 0; i < r; i++){
			subjects =  new ArrayList<List<List<Double>>>(sn);
			for(int k = 0; k < sn; k++){
				classes = new ArrayList<List<Double>>(cn);
				for(int j = 0; j < cn; j++){
					classes.add(new ArrayList<Double>());
				}
				subjects.add(classes);
			}
			rows.add(subjects);
		}
		return rows;
	}
	
	private DefaultBoxAndWhiskerCategoryDataset getTest(){
		DefaultBoxAndWhiskerCategoryDataset classDataset =
			new DefaultBoxAndWhiskerCategoryDataset(); 
		
		List<Double> list = new ArrayList<Double>();
		List<Double> empty = new ArrayList<Double>();
		list.add(4.0);
		list.add(3.0);
		
		classDataset.add(list, 1, 1);
		classDataset.add(empty, 2, 1);
		classDataset.add(list, 3, 1);
				
		classDataset.add(list, 1, 2);
		classDataset.add(empty, 2, 2);
		classDataset.add(list, 3, 2);
		
		classDataset.add(empty, 1, 3);
		classDataset.add(list, 2, 3);
		classDataset.add(list, 3, 3);
		
		classDataset.add(empty, 1, 4);
		classDataset.add(list, 2, 4);
		classDataset.add(list, 3, 4);
		
		return classDataset;
	}
	/**
	 * Create the dataset that is to be used to create a prediction plot
	 * where the x-axis is over session and class. 
	 * @return A dataset where each box and whisker item is denotes a split
	 * and its class, and where the data for that item is composed of the row
	 * means derived from the mPPTrue array.
	 */
	private DefaultBoxAndWhiskerCategoryDataset getSubjectByClassDataset(){
		int[] slMap =  getMapForLabels(mSplitObjLabels);
		int[] clMap = getMapForLabels(mClassLabels);
		int sl;
		int cl;
		int lIndex;
		int rows = mSplitVols.length;
		int cols = mSplitVols[0].length;
		
		//Each value of the mPPTrue array is mapped to a subject and class.
		List<List<List<List<Double>>>> mappedVals = blankArray(rows,slMap.length,clMap.length);
				
		for(int r = 0; r < rows; r++){
			for(int c = 0; c < cols; c++){
				if(mPPTrue[r][c] != -1 && mSplitVols[r][c] > 0){
					lIndex = (int) (mSplitVols[r][c] - 1);
					sl = Arrays.binarySearch(slMap,	(int) mSplitObjLabels[lIndex]);
					cl = Arrays.binarySearch(clMap, (int) mClassLabels[lIndex]);
					Double val = mPPTrue[r][c];
					mappedVals.get(r).get(sl).get(cl).add(val);
				}
			}
		}
		
		/*
		 * We now have the values for each subject and each class for each row.
		 * Now we must find the mean values for each subject and class for each
		 * row. Then we must find the mean of those mean values for each 
		 * subject,class and each row.
		 */
		DefaultBoxAndWhiskerCategoryDataset classDataset =
			new DefaultBoxAndWhiskerCategoryDataset();
		int nSubjects = slMap.length,nClasses = clMap.length;
		int length;
		List<Double> rowvals;
		List<Double> means = new ArrayList<Double>();
		double rowmean;
		double [][][][] savedVals = new double[rows][slMap.length][clMap.length][];
		int i = 0;
		
		for(int subj = 0; subj < nSubjects; subj++){
			for(int cls = 0; cls < nClasses; cls++){
				means.clear();
				i = 0;
				for(int r = 0; r < rows; r++){
					rowvals = mappedVals.get(r).get(subj).get(cls); 
					length = rowvals.size(); 
					if(length > 0){
						
						if(savedVals[r][subj][cls] == null){
							savedVals[r][subj][cls] = new double[length];
						}
						
						i = 0; rowmean = 0;
						for(double v : rowvals){
							rowmean +=v;
							savedVals[r][subj][cls][i++] = v;
						}
						rowmean = rowmean / length;
						means.add(rowmean);
					}
				}
				//if means is empty then it means that for every single row
				//this subject belonging to this class contained no values.
				classDataset.add(means, mClassNames.get(cls), slMap[subj]);
			}
		}
		setSplitByClassVals(savedVals);
		return classDataset;
	}
		
	
	/* precondition: mSplitObjLabels contains values starting at 1, and doesn't skip any numbers.
	 * For example if mSplitObjLabels contains these values {1, 1, 1, 2, 2, 2, 4, 4, 4}
	 * then the method won't work.
	 */
	private DefaultBoxAndWhiskerCategoryDataset getBySplitObjDataset() {	
		DefaultBoxAndWhiskerCategoryDataset classDataset =
				new DefaultBoxAndWhiskerCategoryDataset();
			
		ArrayList<ArrayList<Double>> vals = getBlankValuesArray();
		int numSplitObj = vals.size();
		double[] rowMean = null;
		int[] numAdded = null;
				
		for (int i = 0; i < mSplitVols.length; i++) {
			rowMean = new double[numSplitObj];
			numAdded = new int[numSplitObj];
			for (int j = 0; j < mSplitVols[i].length; j++) {
				int vol = (int) mSplitVols[i][j];
				double data = (double) mPPTrue[i][j];
				int splitObj = 0;
				if (vol > 0 && data >= 0) {
					splitObj = (int) mSplitObjLabels[vol - 1];
					rowMean[splitObj - 1] += data;
					numAdded[splitObj - 1]++;
				}
			}
			
			for (int p = 0; p < numSplitObj; p++) {
				rowMean[p] /= numAdded[p];
				if (!Double.isNaN(rowMean[p]))
					vals.get(p).add(rowMean[p]);
			}
		}

		for (int p = 0; p < numSplitObj; p++) {
			classDataset.add(vals.get(p), 0, p + 1);
		}
		setSplitObjectValues(vals); 
		return classDataset;
	}
		
	private double[][][][] getSplitByClassVals(){
		return mSplitByClassVals;
	}
	
	private void setSplitByClassVals(double[][][][] sbc){
		mSplitByClassVals = sbc;
	}
	
	/**
	 * For each split object calculate the mean of its means.
	 * @return an arraylist containing a single mean value for each split object.
	 */
	private ArrayList<Double> getSplitObjMeans(){
		ArrayList<ArrayList<Double>> vals = getSplitObjValues();
		ArrayList<Double> splitObjMeans = new ArrayList<Double>(vals.size());
		Double temp = 0.0;
		//Calculate each split object's mean.
		for (ArrayList<Double> splitObj : vals){
			if(splitObj.size() == 0){
				splitObjMeans.add(-1.0); //indicator that split obj has no means.
			}
			else{
				for (Double val : splitObj) {
					temp = temp + val;
				}
				splitObjMeans.add(temp/splitObj.size());
				temp = 0.0;
			}
		}
		return splitObjMeans;
	}

	/**
	 * same as getSplitObjMeans() but this time we are looking at medians.
	 * @return the median of the means for each split object.
	 */
	private ArrayList<Double> getSplitObjMedians(){
		ArrayList<ArrayList<Double>> vals = getSplitObjValues();
		ArrayList<Double> splitObjMedian = new ArrayList<Double>(vals.size());

		for (ArrayList<Double> splitObj : vals){
			if(splitObj.size() == 0){
				splitObjMedian.add(-1.0);
			}
			else{

				Double[] sortedVals = new Double[splitObj.size()];
				splitObj.toArray(sortedVals);
				Arrays.sort(sortedVals);
				
				int middleIndex = sortedVals.length / 2;

				if(sortedVals.length % 2 == 0){
					splitObjMedian.add(
							(sortedVals[middleIndex]
                            + sortedVals[middleIndex - 1]) / 2);
				}else{
					splitObjMedian.add(sortedVals[middleIndex]);
				}

			}
		}
		return splitObjMedian;
	}

	//we're using a result file that doesn't include split-object info
	//required to plot prediction by split object
	private DefaultBoxAndWhiskerCategoryDataset getCombinedDataset() {
		DefaultBoxAndWhiskerCategoryDataset classDataset =
				new DefaultBoxAndWhiskerCategoryDataset();
		ArrayList<Double> values = new ArrayList<Double>();			
		
			
		if(mPPTrue == null) {
			mResultFilePaths.get(itemIndex);
			NPairsResultModel model = mRepository.getNpairsModel(
					                     mResultFilePaths.get(itemIndex));
			mPPTrue = model.getPPTrueClass();
		}
		
		//calculate the mean for each split 
		for (int i = 0; i < mPPTrue.length; i++) {			
			double rowMean = 0;
			int numAdded = 0;
			for (int j = 0; j < mPPTrue[0].length; j++) {
				if (mPPTrue[i][j] >= 0) {
					rowMean += mPPTrue[i][j];
					numAdded++;
				}
			}
			rowMean /= numAdded;
			values.add(rowMean);
		}
		//classDataset.add(values, 200, 500);
		classDataset.add(values, 0, 0);
		
		return classDataset;
	}
	
	/** 
	 * Precondition: splitLabels is not null.
	 * @param splitLabels an array containing split object labels.
	 * @return the number of split objects contained in the 
	 * passed in split object labels array. 
	 */
	public static int getNumSplitObjects(double[] splitLabels) {
		
		HashSet<Integer> numbers = new HashSet<Integer>(20);
		
		for(double num : splitLabels){
			numbers.add((int)num);
		}
		return numbers.size();
	}
	
	/** 
	 * @return the number of split objects for this file.
	 */
	private int getNumSplitObjects(){
		return getNumSplitObjects(mSplitObjLabels);
	}
	
	private ArrayList<ArrayList<Double>> getBlankValuesArray() {
		int numSplitObjs = getNumSplitObjects();
		
		ArrayList<ArrayList<Double>> vals;
		vals = new ArrayList<ArrayList<Double>>(numSplitObjs);
		
		
		for (int i = 0; i < numSplitObjs; i++) {
			vals.add(new ArrayList<Double>());
		}
		return vals;
	}
	
	public static ArrayList<ArrayList<Double>> getBlankValuesArray(
			double[] splitLabels) {

		int numSplitObjs = getNumSplitObjects(splitLabels);
		
		ArrayList<ArrayList<Double>> vals;
		vals = new ArrayList<ArrayList<Double>>(numSplitObjs);
		
		
		for (int i = 0; i < numSplitObjs; i++) {
			vals.add(new ArrayList<Double>());
		}
		return vals;
	}

	/**
	 * Add the save buttons which allow the user to save the means, medians
	 * or both of the selected result file to a file.
	 */
	private void addSaveButtons(){
		if(ButtonControl == null){
			ButtonControl = new JPanel();
			ButtonControl.setSize(50, 50);

			saveMeanButton = new JButton("Save Means");
			saveMedianButton = new JButton("Save Medians");
			saveBothButton = new JButton("Save Both");
			regressionButton = new JButton("Load data for Regression Plot");

			saveMeanButton.addActionListener(this);
			saveMedianButton.addActionListener(this);
			saveBothButton.addActionListener(this);
			regressionButton.addActionListener(this);

			ButtonControl.add(saveMeanButton);
			ButtonControl.add(saveMedianButton);
			ButtonControl.add(saveBothButton);
			ButtonControl.add(regressionButton);
			mSubPane.setLayout(new GridLayout(2, 1, 0, 0));
		}
	}

	/**
	 * Adds a listener to mTabs which removes the extra "save means,medians, etc
	 * " buttons from the display when the "combined" tab is selected instead of
	 * the "session" (or other split object) tab.
	 */
	private void addTabsListener(){
		//Register only one listener
		//By default there already exists one to handle selection of the tabs.
		if(mTabs.getMouseListeners().length != 1) return;
		
		mTabs.addMouseListener(new MouseAdapter() {
			@Override
			public void mouseClicked(MouseEvent e) {
				tabClickAction();
			}
		});
	}

	/**
	 * Action performed when a tab is clicked on. If the tab represents 
	 * a combined plot then remove the button control panel for saving
	 * means,medians etc. If it is a plot by split object then make the
	 * button control panel visible. 
	 */
	private void tabClickAction(){
		
		int selectedIndex = mTabs.getSelectedIndex();
		if(selectedIndex >= 1){
			mSubPane.setLayout(new GridLayout(3, 1, 0, 0));
			mSubPane.add(ButtonControl);
			if(selectedIndex == 2){
				regressionButton.setVisible(false);
			}else{
				regressionButton.setVisible(true);
			}
		}else{
			mSubPane.remove(ButtonControl);
			mSubPane.setLayout(new GridLayout(2,1,0,0));
		}
		mSubPane.revalidate();
		mSubPane.repaint();

		chartPersistence.setSelectedTab(getCurrentFile(), selectedIndex);
	}
	
	private void setUpTabs(boolean tripleTabs){
		if(tripleTabs){ //two tab
			if(mTabs == null){
				mTabs = new JTabbedPane();
				mTabs.add("Combined", new ChartPanel(null));
				// TODO: generalize name of panel when prediction plot is
				// displayed by split object (e.g. session)
				mTabs.add("Plot by " + mSplitType, new ChartPanel(null));
				mTabs.add("Plot by " + mSplitType + " & Class", 
						  new ChartPanel(null));
				
				addTabsListener();
				remove(mChartPanel);
				add(mTabs, BorderLayout.CENTER);
			}else{
				mTabs.removeAll();
				mTabs.add("Combined", new ChartPanel(null));
				mTabs.add("Plot by " + mSplitType, new ChartPanel(null));
				mTabs.add("Plot by " + mSplitType + " & Class",
							new ChartPanel(null));
			}
		}else{
			if(mTabs == null){
				mTabs = new JTabbedPane();
				mTabs.add("Combined", new ChartPanel(null));
				addTabsListener();
				remove(mChartPanel);
				add(mTabs, BorderLayout.CENTER);
			}
			else{
				mTabs.removeAll();
				mTabs.add("Combined", new ChartPanel(null));
			}
		}
	}

	private void fillPanels(int fileIndex) {

		mButtonPanel.remove(mLocalButton);

		//if we're using one of the new results with by-split-object prediction
		if (!(mSplitVols == null || mSplitObjLabels == null)) {
			addSaveButtons();
			
			//Are the split obj labels valid? If not make the frames blank.
			if(!checkValidSplitObjLabels(mSplitObjLabels)){
				mPredChartPanel[0] = null;
				mPredChartPanel[1] = null;
				mPredChartPanel[2] = null;
 			}else{
								
				mPredChartPanel[0] = createPlot(fileIndex, 
						getCombinedDataset(),true,false,false);
				mPredChartPanel[1] = createPlot(fileIndex, 
						getBySplitObjDataset(), false,false,false);
				mPredChartPanel[2] = createPlot(fileIndex,
						getSubjectByClassDataset(), false,true,true);
			}
			
			setUpTabs(true);
			mTabs.setComponentAt(0, mPredChartPanel[0]);
			mTabs.setComponentAt(1, mPredChartPanel[1]);
			mTabs.setComponentAt(2, mPredChartPanel[2]);
		} else {
			addSaveButtons();
						
			DefaultBoxAndWhiskerCategoryDataset classDataset0 = getCombinedDataset();
			
			mPredChartPanel[0] = createPlot(fileIndex, classDataset0, true,false,false);

			setUpTabs(false);
			mTabs.setComponentAt(0, mPredChartPanel[0]);
		}
		
	}	
	
	/**
	 * All split object labels must be a number that is <= the number of
	 * split object labels. Further more the split object labels must be
	 * listed sequentially from least to greatest.
	 * @return true if this the case, false otherwise. true also if split
	 * labels happens to be null.
	 */
	private boolean checkValidSplitObjLabels(double[] splitLabels){
		if(splitLabels == null) return true;
		
		int greatest = -1;
		int label;
		int numLabels = getNumSplitObjects(splitLabels);
		
		
		for(double value : splitLabels){
			label = (int) value;
			greatest = (greatest > label) ? greatest : label;
			
			//sequential rule broken.
			if(label < greatest) return false;
			
			//the label number is greater than the number of labels.
			if(label > numLabels) return false;
		
		}
		return true;
	}
	/**
	 * Initialize persistence for a single file. This needs to be run once for
	 * a file before any charts for it are drawn.
	 * @param filename the file in question.
	 */
	private void initChartPersistence(String filename){
		if(!chartPersistence.containsFile(filename)){
			chartPersistence.addNewFile(filename);
			ArrayList<ArrayList<JFreeChart>> panels =
					chartPersistence.getCharts(filename);

			panels.add(new ArrayList<JFreeChart>()); // tab 1 'combined dataset'
			panels.add(new ArrayList<JFreeChart>()); // tab 2 'subject dataset'

			panels.get(0).add(null);
			panels.get(1).add(null);

		}
	}

	private ChartPanel createPlot(int fileIndex, 
			DefaultBoxAndWhiskerCategoryDataset classDataset,
			boolean isCombined, boolean legend, boolean useSpecialRenderer) {

		NPairsResultModel model;
		//i'm positive these 4 lines don't do anything but I'm not taking them
		//out because I don't want things to explode just in case...
		model = mRepository.getNpairsModel(mResultFilePaths.get(fileIndex) );
		mPPTrue = model.getPPTrueClass();
		mSplitObjLabels = model.getSplitObjLabels();
		mSplitVols = model.getSplitVols();
		

		CategoryPlot categoryPlot;
		JFreeChart predictionPlot;
		CategoryAxis categoryAxis;
		NumberAxis valueAxis;
		AbstractCategoryItemRenderer renderer;

		//Persistence information. See comment at the beginning of the class.
//		String curFile = getCurrentFile();
//		int tabIndex = 0;

		//the prediction plot has only two tabs with index 0 or 1. combined
		//tab belongs to index 0.
//		if(isCombined) tabIndex = 1;

//		predictionPlot = chartPersistence.getChart(curFile, tabIndex);
//
//		if (predictionPlot != null) {
//			categoryPlot = (CategoryPlot) predictionPlot.getPlot();
//			categoryPlot.setDataset(classDataset);
//
//			ValueAxis va = categoryPlot.getRangeAxis();
//			if(mGlobalButton.isSelected()){
//				va.setRange(new Range(0, 1.0));
//			}else{ va.setAutoRange(true); }
//
//			return new ChartPanel(predictionPlot);
//		}

		/*Create new chart and panel*/
		// If we are creating a combined data set, set the title of the
		//horizontal axis to the current file name instead of "sessions"
		//(or whatever current split object is).
		if (isCombined){
			categoryAxis = new CategoryAxis(getCurrentFile());
			categoryAxis.setTickLabelsVisible(false);
		} else {
			categoryAxis = new CategoryAxis(mSplitType + "s");
			categoryAxis.setTickLabelsVisible(true);
		}

		valueAxis = new NumberAxis("Posterior Probability");
		valueAxis.setAutoRangeIncludesZero(false);
		
		if(useSpecialRenderer){
			renderer = new PlotBySplitClassRenderer();
		}else{
			renderer = new NpairsBoxAndWhiskerRenderer();
		}
		renderer.setBaseToolTipGenerator(new BoxAndWhiskerToolTipGenerator());

		
		categoryPlot = new CategoryPlot(classDataset, categoryAxis,
				valueAxis, renderer);
		predictionPlot = new JFreeChart("Prediction",
				JFreeChart.DEFAULT_TITLE_FONT, categoryPlot, legend);
		
//		chartPersistence.setChart(curFile, tabIndex, predictionPlot);

		if (mGlobalButton.isSelected()) {
			valueAxis.setRange(new Range(0, 1.0) );
		}else{
			NpairsBoxWhiskerAxisFix.fixBoxAndWhiskerRange(categoryPlot);
		}
	
		return new ChartPanel(predictionPlot);
	}
}
