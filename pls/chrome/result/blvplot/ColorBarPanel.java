package pls.chrome.result.blvplot;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.text.DecimalFormat;
import java.util.ArrayList;

import javax.swing.ImageIcon;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SpringLayout;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.statistics.HistogramDataset;
import org.jfree.ui.RectangleInsets;

import pls.chrome.result.model.BrainData;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.ResultModel;
import pls.shared.GlobalVariablesFunctions;

@SuppressWarnings("serial")
/**
 * This class represents the histogram chart you see above the plotted brain
 * images.
 */
public class ColorBarPanel extends JPanel{
	private static int DEFAULT_BAR_WIDTH = 350;

	private String mResultFilename = "";
	private String mDataTypeName = "";
	private int mLvNum = -1;
	private int mLagNum = -1;
	
	private ColorGradient mColGrad;
	private ColorBar mColBar;
	private ChartPanel mHistogram = null;
	
	private GeneralRepository mRepository = null;
	
	SpringLayout mLayout = new SpringLayout();
	
	// Some Formatting patterns to always have the same maximum number of
	// digits and same minimum precision.
	static String exponentialPattern = "0.00E0";
	static String normalPattern1 = "0.0###";
	static String normalPattern2 = "00.0##";
	static String normalPattern3 = "000.0#";
	static String normalPattern4 = "0000.0";
	
	static DecimalFormat mFormatter;
	
	public ColorBarPanel(GeneralRepository repository) {
		mRepository = repository;
		
		setLayout(mLayout);
		
		mFormatter = (DecimalFormat)DecimalFormat.getInstance();
		
		// Initializes the histogram
		mHistogram = new ChartPanel(null);
		mHistogram.setPreferredSize(new Dimension(350, 35) );
		
		setPreferredSize(new Dimension(500, 75));
//		setMinimumSize(new Dimension(350, 200));
//		createColourBar();
		
		// Add the items
		add(mHistogram);
		
		GeneralRepository.setColorBarPanel(this);
	}
	
	private void setupTicksAndLabels() {
		double[] colourScale = mRepository.getGlobalColourScale();
		
		if (!mRepository.getUseGlobalScale() ) {
			colourScale = mRepository.getGeneral().getBrainData().getColourScaleModel().getColourScale();
		}
		
		double max = colourScale[0], min = colourScale[1];
		
		// Create the tick marks
		BufferedImage notch = new BufferedImage(1, 5, BufferedImage.TYPE_BYTE_BINARY);
		Graphics2D g2 = notch.createGraphics();
		g2.setColor(Color.BLACK);
		g2.drawRect(0, 0, 1, 5);
		
		// Create the zero spot indicator
		BufferedImage zeroMark = new BufferedImage(1, 20, BufferedImage.TYPE_BYTE_BINARY);
		g2 = zeroMark.createGraphics();
		g2.setColor(Color.BLACK);
		g2.drawRect(0, 0, 1, 20);
		
		// Create the always existing marks
		JLabel maxBarLabel = new JLabel(new ImageIcon(notch) );
		JLabel zeroBarLabel = new JLabel(new ImageIcon(zeroMark) );
		JLabel minBarLabel = new JLabel(new ImageIcon(notch) );
		
		// Calculate some useful values
		double totalSize = max - min;
		double zeroRatio = 1 - (max / totalSize);
		
		// Create the strings that will be used in the max, zero and min labels
		String maxText = formatNumber(max);
		String zeroText = formatNumber(0.0);
		String minText = formatNumber(min);
		
		// Create the max, zero and min labels
		JLabel maxLabel = new JLabel(maxText);
		JLabel zeroLabel = new JLabel(zeroText);
		JLabel minLabel = new JLabel(minText);
		
		int zeroPosition = (int)(mColBar.getPreferredSize().width * zeroRatio);
		
		// Add our dudes
		add(maxBarLabel);
		mLayout.putConstraint(SpringLayout.EAST, maxBarLabel, 0, SpringLayout.EAST, mColBar);
		mLayout.putConstraint(SpringLayout.NORTH, maxBarLabel, 0, SpringLayout.SOUTH, mColBar);
		
		add(maxLabel);
		int labelWidth = maxLabel.getPreferredSize().width;
		mLayout.putConstraint(SpringLayout.WEST, maxLabel, -labelWidth/2, SpringLayout.WEST, maxBarLabel);
		mLayout.putConstraint(SpringLayout.NORTH, maxLabel, 0, SpringLayout.SOUTH, maxBarLabel);
		
		add(zeroBarLabel);
		mLayout.putConstraint(SpringLayout.WEST, zeroBarLabel, zeroPosition, SpringLayout.WEST, mColBar);
		mLayout.putConstraint(SpringLayout.NORTH, zeroBarLabel, 0, SpringLayout.NORTH, mColBar);
		
		add(zeroLabel);
		labelWidth = zeroLabel.getPreferredSize().width;
		mLayout.putConstraint(SpringLayout.WEST, zeroLabel, -labelWidth/2, SpringLayout.WEST, zeroBarLabel);
		mLayout.putConstraint(SpringLayout.NORTH, zeroLabel, 0, SpringLayout.SOUTH, zeroBarLabel);
		
		add(minBarLabel);
		mLayout.putConstraint(SpringLayout.WEST, minBarLabel, 0, SpringLayout.WEST, mColBar);
		mLayout.putConstraint(SpringLayout.NORTH, minBarLabel, 0, SpringLayout.SOUTH, mColBar);

		add(minLabel);
		labelWidth = minLabel.getPreferredSize().width;
		mLayout.putConstraint(SpringLayout.WEST, minLabel, -labelWidth/2, SpringLayout.WEST, minBarLabel);
		mLayout.putConstraint(SpringLayout.NORTH, minLabel, 0, SpringLayout.SOUTH, minBarLabel);
	}
	
	/**
	 * Given a double, formats it to a maximum number of digits.  Converts to
	 * scientific notation if necessary.
	 */
	public static String formatNumber(double number) {
		double absNumber = Math.abs(number);
		
		if (number == 0.0) {
			mFormatter.applyPattern(normalPattern1);
		}
		else if (absNumber < 0.01 || absNumber >= 10000.0) {
			mFormatter.applyPattern(exponentialPattern);
		}
		else if (absNumber >= 1000.0) {
			mFormatter.applyPattern(normalPattern4);
		}
		else if (absNumber >= 100.0) {
			mFormatter.applyPattern(normalPattern3);
		}
		else if (absNumber >= 10.0) {
			mFormatter.applyPattern(normalPattern2);
		}
		else {
			mFormatter.applyPattern(normalPattern1);
		}
		
		return mFormatter.format(number);
	}
	
	/**
	 * Create a histogram based on the current values in mBrainData
	 */
	private void createHistogram() {
		ResultModel model = mRepository.getGeneral();
		BrainData bData = model.getBrainData();
		double[] colourScale = mRepository.getGlobalColourScale();
		
		if (!mRepository.getUseGlobalScale() ) {
			colourScale = bData.getColourScaleModel().getColourScale();
		}
		
		double max = colourScale[0], min = colourScale[1];
		
		HistogramDataset dataset = new HistogramDataset();
		
		ArrayList<Double> someDoubles = new ArrayList<Double>();
		
//		boolean OK = true;
		for (double[] lagBlock : bData.getData().values() ) {
//			if (Double.isNaN(lagBlock[mLagNum])) {
//				OK = false;
//			}
			someDoubles.add(lagBlock[mLagNum]);
		}
//		if (!OK) {
//			GlobalVariablesFunctions.showErrorMessage("WARNING: Brain image results volume contains NaNs.");
//		}
		
		if (someDoubles.size() > 0) {
			double[] allData = new double[someDoubles.size()];
			
			for (int i = 0; i < someDoubles.size(); ++i) {
				allData[i] = someDoubles.get(i);
			}
			
			try {
				dataset.addSeries("yay", allData, 128);
				JFreeChart myChart = ChartFactory.createHistogram(null, null, null, dataset, PlotOrientation.VERTICAL, false, false, false);
				XYPlot plot = myChart.getXYPlot();
				plot.getDomainAxis().setVisible(false);
				plot.getRangeAxis().setVisible(false);
				plot.getDomainAxis().setLowerBound(min);
				plot.getDomainAxis().setUpperBound(max);
				mHistogram.setChart(myChart);
				
				removeAll();
				add(mHistogram);
				mLayout.putConstraint(SpringLayout.WEST, mHistogram, 50, SpringLayout.WEST, this);
				
				RectangleInsets ri = myChart.getXYPlot().getInsets();
				
				createColourBar();
				
				mLayout.putConstraint(SpringLayout.WEST, mColBar, (int)ri.getLeft() + 1, SpringLayout.WEST, mHistogram);
				mLayout.putConstraint(SpringLayout.NORTH, mColBar, 0, SpringLayout.SOUTH, mHistogram);
			}
			catch (Exception e) {
				GlobalVariablesFunctions.showErrorMessage("Unable to create the histogram.");
			}
		}
		else {
			mHistogram.setChart(null);
		}
	}

	public void refreshPanel() {
//		removeAll();
		
		ResultModel model = mRepository.getGeneral();
		String currResultFile = mRepository.getSelectedResultFile();
		String currDataType = model.getSelectedDataType();
		int currLv = model.getBrainData().getLv();
		int lagNum = model.getSelectionModel().getSelectedVoxel()[3];
		
//		if (!currResultFile.equals(mResultFilename) ||
//				!currDataType.equals(mDataTypeName) ||
//				currLv != mLvNum ||
//				lagNum != mLagNum) {
			mResultFilename = currResultFile;
			mDataTypeName = currDataType;
			mLvNum = currLv;
			mLagNum = lagNum;
			createHistogram();
//		}
	}

	private void createColourBar() {
		RectangleInsets ri = mHistogram.getChart().getXYPlot().getInsets();
		
		int width = DEFAULT_BAR_WIDTH - (int)ri.getLeft() - (int)ri.getRight();
		
		double[] colourScale = mRepository.getGlobalColourScale();
		if (!mRepository.getUseGlobalScale() ) {
			colourScale = mRepository.getGeneral().getBrainData().getColourScaleModel().getColourScale();
		}
		
		mColGrad = new ColorGradient(colourScale[0], colourScale[1], colourScale[2]);
		mColBar = new ColorBar(mColGrad, width, 15);

		add(mColBar);
		
		setupTicksAndLabels();
	}
}
