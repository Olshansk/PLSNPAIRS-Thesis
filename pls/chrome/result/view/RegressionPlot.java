package pls.chrome.result.view;

import java.awt.BorderLayout;
import java.awt.event.ActionEvent;
import java.util.ArrayList;
import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.DefaultXYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.shared.ImageWriter;
import pls.shared.MLFuncs;

/**
 * The plot that appears when "Load regression data" is clicked on
 * in PredictionPlot.java (The prediction plot).
 */
public class RegressionPlot extends AttachDetachOption{
	
	private ChartPanel cPanel = null; // panel holding the plot

	private JButton destroyPlotButton; //button that closes the plot
	private JRadioButton meanButton; // x-axis selection buttons
	private JRadioButton medianButton;
	private JCheckBox subjDisplay;

	//when a window is closed it is actually hidden and detached.
	//when we reopen the window we want to know whether this window was
	//currently detached or attached
	private boolean curDetached;
	private boolean closed;
	private boolean meansSelected = true; //means selected for x-axis? default is true.
	
	//loaded x-axis/y-axis values.
	private double[] fixedMeans;
	private double[] fixedMedians;
	private double[] fixedMeansYAxis;
	private double[] fixedMediansYAxis;
	
	//Mappings that keep track of which index in fixed{Means,Medians} belongs
	//to which subject.
	private ArrayList<Integer> subjMappingMeans = new ArrayList<Integer>();
	private ArrayList<Integer> subjMappingMedians = new ArrayList<Integer>();
	
	//The type of split (run, session etc, used for the labels).
	private String splitType;
	
	public RegressionPlot(String title, GeneralRepository repository){
		
		super(repository,title);
		mResultFilePaths = new ArrayList<String>();
		mResultFilePaths.add("hi");
		curDetached = false;
		closed = true;		
	}

	public void initialize(){}

	/**
	 * Record that this panel has been detached when the the detach icon is
	 * clicked on [mTabbedPane (in PlotManager)].
	 * @param val boolean value
	 */
	public void setcurDetached(boolean val){
		curDetached = val;
	}

	public void doSaveAs(){
		ImageWriter.doSaveAs(cPanel);
	}
	
	@Override
	public void actionPerformed(ActionEvent e){
		if(e.getSource() == destroyPlotButton){
			closePlot();
		}else if(e.getSource() == meanButton){
			meansSelected = true;
			fastPlotRedraw();
		}else if(e.getSource() == medianButton){
			meansSelected = false;
			fastPlotRedraw();
		}else if(e.getSource() == mAttachDetachButton){
			
			super.actionPerformed(e);
			if (mAttachDetachButton.getText().equals("Detach"))
				curDetached = false;
			else
				curDetached = true;
		}else if(e.getSource() == subjDisplay){
		  XYPlot plot = (XYPlot) cPanel.getChart().getPlot();

		  if(subjDisplay.isSelected()){
			plot.getRenderer(0).setBaseItemLabelsVisible(true);
		  }else{
			plot.getRenderer(0).setBaseItemLabelsVisible(false);
		  }
		}
		else{
			super.actionPerformed(e);
		}
	}

	private XYSeriesCollection createDataset(double[] xAxis, double[] yAxis,
			ArrayList<Integer> mapping){

	  XYSeries points = new XYSeries(0);
	  XYSeriesCollection dataset = new XYSeriesCollection(points);

	  for(int i = 0; i < xAxis.length; i++){
		RegressionDataItem point = new RegressionDataItem(xAxis[i],
				yAxis[i], mapping.get(i), splitType);
		points.add(point);
	  }

	  return dataset;
	}

	/**
	 * Creates the option to test the regressionplot with mock data.
	 * Call this instead of drawPlot in prediction plot if you want to test
	 * this class.
	 */
	void testData(){
	  testRegressionPlot tester = new testRegressionPlot(this);
	}
	
	/**
	 * Called from predictionPlot when the "Load regression values" button
	 * is clicked. Draw the regression plot.
	 * @param rVals the loaded regression values for the y-axis.
	 * @param subjMeans the selected result file in the prediction plot's means.
	 * @param subjMedians same as subjMeans, just the medians this time.
	 * @param resultFN the selected result file's name.
	 * @param loadedFN the file name from where the regression data derived from
	 */
	void drawPlot(ArrayList<Double> rVals,
			ArrayList<Double> subjMeans,
			ArrayList<Double> subjMedians,
			String resultFN,
			String loadedFN,
			String splitType){
		
		this.splitType = splitType;
		if(splitType == null) this.splitType = "Split Object";
		
		mDetachedFrame.setTitle("Regression Plot");
		if(!curDetached && closed){
			attachWindow(); //attach again.
			closed = false;
		}else if(curDetached && closed){
			mDetachedFrame.setVisible(true);
			closed = false;
		}
		
		/*Get plot data*/
		double[][] fixedVals;
		ArrayList<Integer> mapping;

		/*Saved the parsed values for later when the user may need to use
		 * them again if they select a different x-axis through the radio
		 * buttons*/
		fixedVals = fixData(rVals, subjMeans,subjMappingMeans);
		fixedMeans = fixedVals[0];
		fixedMeansYAxis = fixedVals[1];

		fixedVals = fixData(rVals, subjMedians,subjMappingMedians);
		fixedMedians = fixedVals[0];
		fixedMediansYAxis = fixedVals[1];

		if(meansSelected){
			fixedVals[0] = fixedMeans;
			fixedVals[1] = fixedMeansYAxis;
			mapping = subjMappingMeans;
		}
		else{
			fixedVals[0] = fixedMedians;
			fixedVals[1] = fixedMediansYAxis;
			mapping = subjMappingMedians;
		}

		/*Plot the points*/
		XYPlot plot = new XYPlot();
		
		XYSeriesCollection dataset = createDataset(fixedVals[0],
				fixedVals[1],mapping);

		XYItemRenderer renderer1 = new XYLineAndShapeRenderer(false, true);
		ValueAxis domain = new NumberAxis(resultFN);
		ValueAxis range = new NumberAxis(loadedFN);

		//if there is only a single value to be plotted we need to manually
		//set the range and domain of the axis. 
		if(fixedVals[0].length == 1){
			domain.setRangeAboutValue(fixedVals[0][0], 1.0);
			range.setRangeAboutValue(fixedVals[1][0], 1.0);
		}

		renderer1.setBaseItemLabelsVisible(true);
		renderer1.setBaseItemLabelGenerator(new RegressionItemGenerator());
		((NumberAxis) domain).setAutoRangeIncludesZero(false);
		((NumberAxis) range).setAutoRangeIncludesZero(false);
	
		plot.setDataset(0, dataset);
		plot.setRenderer(0, renderer1);
		plot.setDomainAxis(0, domain);
		plot.setRangeAxis(0, range);
		plot.mapDatasetToDomainAxis(0, 0); //map dataset 0 to domain 0
		plot.mapDatasetToRangeAxis(0, 0);

		/*Plot the line*/
		DefaultXYDataset lineset = new DefaultXYDataset();
		lineset.addSeries(0, calculateOLSFit(fixedVals));

		XYItemRenderer renderer2 = new XYLineAndShapeRenderer(true, false);

		plot.setDataset(1, lineset);
		plot.setRenderer(1, renderer2);
		plot.mapDatasetToDomainAxis(1, 0); //map dataset 0 to domain 0
		plot.mapDatasetToRangeAxis(1, 0);

		JFreeChart chart = new JFreeChart(plot);
		chart.removeLegend();

		//lay the widgets out for the chart panel only once.
		if(cPanel == null) layoutWidgets(chart);
		else cPanel.setChart(chart);
		
		//Change the text if we are loading in new data because it may
		//have a different split type. i.e previous data loaded was session
		//and now we are dealing with runs.
		subjDisplay.setText("Display " + this.splitType + " Labels");
	}

	/**
	 * Redraws the plot in response to a radio button selection.
	 */
	private void fastPlotRedraw(){
		double[][] data = new double[2][];
		ArrayList<Integer> mapping;

		if(meansSelected){
		  data[0] = fixedMeans;
		  data[1] = fixedMeansYAxis;
		  mapping = subjMappingMeans;
		}
		else{
		  data[0] = fixedMedians;
		  data[1] = fixedMediansYAxis;
		  mapping = subjMappingMedians;
		}

		/*Update points*/
		XYPlot plot = (XYPlot) cPanel.getChart().getPlot();
		XYSeriesCollection dataset = createDataset(data[0],data[1],mapping);
		plot.setDataset(0, dataset);
		
		/*Update line of best fit*/
		DefaultXYDataset lineset = (DefaultXYDataset) plot.getDataset(1);
		lineset.removeSeries(0);
		lineset.addSeries(0, calculateOLSFit(data));
	}

	/**
	 * Lay out the widgets needed to interact with the chart.
	 * Lay out the chart as well.
	 * @param chart the chart to be displayed.
	 */
	private void layoutWidgets(JFreeChart chart){
		cPanel = new ChartPanel(chart);

		JPanel buttonPanel = new JPanel();
		JPanel rButtonPanel = new JPanel();
		JPanel controlPanel = new JPanel(new BorderLayout());
		ButtonGroup rGroup = new ButtonGroup();

		meanButton = new JRadioButton("Means");
		medianButton = new JRadioButton("Medians");
		subjDisplay = new JCheckBox("Display " + splitType + " Labels");

		destroyPlotButton = new JButton("Close Plot");
		destroyPlotButton.addActionListener(this);

		meanButton.setSelected(true);
		meanButton.addActionListener(this);
		medianButton.addActionListener(this);
		subjDisplay.setSelected(true);
		subjDisplay.addActionListener(this);

		rGroup.add(meanButton);
		rGroup.add(medianButton);
		
		rButtonPanel.add(meanButton);
		rButtonPanel.add(medianButton);
		rButtonPanel.add(subjDisplay);

		buttonPanel.add(mAttachDetachButton);
		buttonPanel.add(destroyPlotButton);
		
		controlPanel.add(rButtonPanel, BorderLayout.NORTH);
		controlPanel.add(buttonPanel, BorderLayout.SOUTH);

		this.setLayout(new BorderLayout());
		add(controlPanel, BorderLayout.SOUTH);
		add(cPanel, BorderLayout.CENTER);

		//mDetachedFrame.setVisible(false);
	}

	/**
	 * Calculates the line of best fit.
	 * @param fixedVals the x-axis, y-axis data for which the line of best fit
	 * is drawn for.
	 * @return the coordinates for the line of best fit.
	 */
	private double[][] calculateOLSFit(double[][] fixedVals){
		double[] X = fixedVals[0];
		double[] Y = fixedVals[1];
		double corr = MLFuncs.corr(X, Y);
		double stdx = MLFuncs.std(X);
		double stdy = MLFuncs.std(Y);

		double b = corr * stdy / stdx;
		double a = MLFuncs.avg(Y) - b*MLFuncs.avg(X);

		//Calculate Y = bX + a.
		double[] newX = new double[X.length];
		double[] newY = new double[X.length];
		double[][] retval = new double[2][];

		for(int i = 0; i < X.length; i++){
			newY[i] = b*X[i] + a;
			newX[i] = X[i];
		}

//		System.out.println("CORR IS : " + corr);
//		System.out.println("STDX: " + stdx);
//		System.out.println("STDY: " + stdy);
//		System.out.println("AVG Y: " + MLFuncs.avg(Y));
//		System.out.println("AVG X: " + MLFuncs.avg(X));
//		System.out.println("B IS: " + b);
//		System.out.println("A IS: " + a);
		retval[0] = newX;
		retval[1] = newY;
		return retval;
	}

	/**
	 * Given the data provided from prediction plot, correct it such that
	 * for any elements on the x-axis that register -1, or for any elements on
	 * the y-axis that register null, the corresponding value in the other
	 * axis becomes void as well. Allow vectors of differing size but truncate
	 * such that all values that index past min(v1.length,v2.length) are 
	 * discared. Optionally warn the user that this truncation is occuring. 
	 * @param yAxis y-axis data.
	 * @param xAxis x-axis data.
	 * @return the fixed axis data.
	 */
	private double[][] fixData(ArrayList<Double> yAxis,ArrayList<Double> xAxis,
			ArrayList<Integer> mapping){

		int min = Math.min(yAxis.size(),xAxis.size());
		double[] xAxisFixed = new double[min];
		double[] yAxisFixed = new double[min];
		double[][] retval = new double[2][];
		mapping.clear();

		int count = 0;
		for (int i = 0; i < min; i++){
			if(yAxis.get(i) != null && xAxis.get(i) != -1){
				xAxisFixed[count] = xAxis.get(i);
				yAxisFixed[count] = yAxis.get(i);
				count++;
				mapping.add(i+1);
			}
		}

		double[] ret1 = new double[count];
		double[] ret2 = new double[count];

		for(int i = 0; i < count; i++){
			ret1[i] = xAxisFixed[i];
			ret2[i] = yAxisFixed[i];
		}

		//Optional message.
//		if(yAxis.size() != xAxis.size()){
//			JOptionPane.showMessageDialog(this,"The two Axis did not contain " +
//					"the same" + "number of data points. Omitting points" +
//					"in the range of min(xAxis,yAxis):max(xAxis,yAxis) ");
//		}

		retval[0] = ret1;
		retval[1] = ret2;
		return retval;
	}

	/**
	 * simulate erasure of the plot by detaching it and making it invisible.
	 */
	private void closePlot(){
		if(mAttachDetachButton.getText().equals("Detach")){
			//hideMe();
			mManager.detachPlot(this);
			curDetached = false;
		}else{
			curDetached = true;			
		}
		mDetachedFrame.setVisible(false);
		closed = true;
	}


}
