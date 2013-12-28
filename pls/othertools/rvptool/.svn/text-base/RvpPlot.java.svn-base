package pls.othertools.rvptool;

import java.awt.BorderLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextPane;
import javax.swing.filechooser.FileFilter;

import npairs.io.NpairsjIO;

import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.Range;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import pls.shared.MLFuncs;

import com.jmatio.io.MatFileFilter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLStructure;

import extern.NewMatFileReader;

@SuppressWarnings("serial")
public class RvpPlot extends JFrame implements ActionListener {
	
	private String mErrorString = "";
	private double mReprodMedian = -1;
	private double mPpTrueClassMedian = -1;
	private double mPredPercentAccuracyMedian = -1;
	boolean mPlotPercentAccuracy = false; // if true, plot prediction percent accuracy instead of
	                                      // posterior probability on Y axis
	
	private XYPlot mPlot = null;
	private JCheckBox mShowPcLabels = new JCheckBox("Show pc labels", true);
	private JButton mSaveValues = new JButton("Save values");
	
	private HashMap<String, HashMap<Integer, double[]>> mCalculatedValues
			= new HashMap<String, HashMap<Integer, double[]>>();
	
	public RvpPlot(ArrayList<CurveGroup> setupInfo, String predType) {
		super("RVP Plot");
		if (predType.equals("Percent Accuracy")) {
			mPlotPercentAccuracy = true;
		}
			
		setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		
		XYSeriesCollection mySeries = new XYSeriesCollection();
		
		double lowest = 0;
		
		for (CurveGroup group : setupInfo) {
			String generalFilename = group.getFilename();
			ArrayList<String> numbers = group.getParsedNumbers();
			
			XYSeries curve = new XYSeries(group.getLabel(), false);
			
			for (String i : numbers) {
				String specificFilename = generalFilename.replace("$", i);
				
				int cvDim = group.getCurveUnit() - 1;
				
				// If no values could be extracted, skip this plot.
				if (!extractValues(group, specificFilename, cvDim) ) {
					continue;
				}
				
				// Represents each point as its own plot so that it can
				// be displayed in the legend as an individual item.
//				XYSeries singlePoint = new XYSeries(i, false);
//				singlePoint.add(mReprodMedian, mPpTrueClassMedian);
				
				if (mReprodMedian < lowest) {
					lowest = mReprodMedian;
				}
				// Adds the point to the main plot used as the actual plot.
				if (mPlotPercentAccuracy) {
					curve.add(new RvpDataItem(mReprodMedian, mPredPercentAccuracyMedian, removePaddedZeros(i) ) );
				}
				else {
					curve.add(new RvpDataItem(mReprodMedian, mPpTrueClassMedian, removePaddedZeros(i) ) );
				}
				
//				mySeries.addSeries(singlePoint);
			}

			//if all the result files in the group fail to load present an error
			if (curve.getItemCount() > 0) {
				mySeries.addSeries(curve);
			} else {
				mErrorString += "No data could be loaded for "
						     + group.getLabel() + ".  Skipping curve.\n";
			}
		}
		
		if (mySeries.getSeriesCount() > 0) {
			NumberAxis domainAxis = new NumberAxis("Reproducibility");
			domainAxis.setAutoRange(false);
			domainAxis.setRange(new Range(lowest - 0.05, 1.0) );
			
			
			NumberAxis valueAxis = new NumberAxis("Prediction");
			if (mPlotPercentAccuracy) {
				valueAxis = new NumberAxis("Prediction Percent Accuracy / 100  ");
			}
			valueAxis.setAutoRange(false);
			valueAxis.setRange(new Range(0.0, 1.0) );
			
			RvpItemLabelGenerator generator = new RvpItemLabelGenerator();
			
			XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
			renderer.setBaseItemLabelsVisible(true);
			renderer.setBaseItemLabelGenerator(generator);
			
			mPlot = new XYPlot(mySeries, domainAxis, valueAxis, renderer);
	
			JFreeChart chart = new JFreeChart("Reproducibility vs Prediction", mPlot);
			
			add(new ChartPanel(chart) );
			
			JPanel southPanel = new JPanel();
			southPanel.add(mShowPcLabels);
			mShowPcLabels.setActionCommand("toggle pc labels");
			mShowPcLabels.addActionListener(this);
			southPanel.add(mSaveValues);
			mSaveValues.setActionCommand("save values");
			mSaveValues.addActionListener(this);
			
			add(southPanel, BorderLayout.SOUTH);
			
			setVisible(true);
			setSize(800, 600);
		} else {
			mErrorString += "Plot not displayed because no curves could be loaded.\n";
		}

		//if anything went wrong present the user with the errors raised.
		if (!mErrorString.equals("") ) {
			JFrame errorFrame = new JFrame("Error log");
			JTextPane errorPane = new JTextPane();
			errorPane.setText(mErrorString);
			errorFrame.add(new JScrollPane(errorPane) );
			
			errorFrame.setVisible(true);
			errorFrame.setSize(640, 480);
		}
	}
	
	private boolean extractValues(CurveGroup group, String filename, int cvDim) {
		double[][] reprod_cc = null;
		double[][] pp_true_class = null;
		double[][] correct_pred = null;
		
		if (filename.endsWith("NPAIRSJresult.mat") ) {
			Map<String, MLArray> resultInfo = null;
			try {
				 resultInfo = new NewMatFileReader(filename,
						new MatFileFilter(new String[]{"npairs_result",
						                               "reprod_cc", 
													   "pp_true_class"}) )
													   .getContent();
			} catch (IOException e) {
				mErrorString += "Couldn't open file: " + filename + "\n";
				return false;
			}
			
			MLArray npairs_result = resultInfo.get("npairs_result");
			MLStructure struct_npairs_result = (MLStructure) npairs_result;
			
			//Check that column of this array signifies the number of cv dims. (the length).
			reprod_cc = ((MLDouble) struct_npairs_result.getField("reprod_cc")).getArray();
			
			if(reprod_cc[0].length < group.getCurveUnit()){
				mErrorString += "Curve (" + group.getLabel() 
				+") specifed a nonexistent CV (" + group.getCurveUnit() +
				")" + " for file " + filename + "\n";
				return false;
			}
						
			MLArray prediction = struct_npairs_result.getField("prediction");
			MLStructure struct_prediction = (MLStructure) prediction;
			MLArray priors = struct_prediction.getField("priors");
			MLStructure struct_priors = (MLStructure) priors;
			
			if (mPlotPercentAccuracy) {
				correct_pred = ((MLDouble) struct_priors.getField("correct_pred")).getArray();
			}
			else {
				pp_true_class = ((MLDouble) struct_priors.getField("pp_true_class")).getArray();
			}
		} else {
			if (mPlotPercentAccuracy) {
				mErrorString = "Prediction percent accuracy data can only be loaded from NPAIRS result " +
					".mat file.";
				return false;
			}
			String ccFile = filename + ".CVA.SUMM.CC";
			String ppFile = filename + ".CVA.SUMM.PP.ppTruePriors";
			
			// Read in the cc file
			try {
				BufferedReader br = new BufferedReader(new FileReader(ccFile) );
				
				String line = br.readLine();
				
				// These are the dimensions
				if (line != null) {
					String[] dims = line.split(" ");
					//number of columns here tells us the number of cv dims (first value parsed) 
					reprod_cc = new double[Integer.parseInt(dims[1])][Integer.parseInt(dims[0])];
					
					if(reprod_cc[0].length < group.getCurveUnit()){
						mErrorString += "Curve (" + group.getLabel() 
						+") specifed a nonexistent CV (" + group.getCurveUnit()
						+ ")" +	" for file " + filename + "\n";
						return false;
					}
				}
				
				line = br.readLine();
				int i = 0;
				
				while (line != null) {
					String[] strings = line.split(" ");

					for (int j = 0; j < strings.length; ++j) {
						reprod_cc[i][j] = Double.parseDouble(strings[j]);
					}
					
					line = br.readLine();
					++i;
				}
			} catch (FileNotFoundException fnfex) {
				mErrorString += "Couldn't find the file: " + ccFile + "\n";
				return false;
			} catch (IOException ioex) {
				mErrorString += "An error occurred while reading open file: " + ccFile + "\n";
				return false;
			}
			
			// Read in the pp file
			try {
				BufferedReader br = new BufferedReader(new FileReader(ppFile) );
				
				String line = br.readLine();
				
				// These are the dimensions
				if (line != null) {
					String[] dims = line.split(" ");
					
					pp_true_class = new double[Integer.parseInt(dims[1])][Integer.parseInt(dims[0])];
				}
				
				line = br.readLine();
				int i = 0;
				
				while (line != null) {
					String[] strings = line.split(" ");
					
					for (int j = 0; j < strings.length; ++j) {
						pp_true_class[i][j] = Double.parseDouble(strings[j]);
					}
					
					line = br.readLine();
					++i;
				}
			} catch (FileNotFoundException fnfex) {
				mErrorString += "Couldn't open file: " + ppFile + "\n";
				return false;
			} catch (IOException ioex) {
				mErrorString += "An error occurred while reading open file: " + ppFile + "\n";
				return false;
			}
		}
		
		if (reprod_cc != null && (pp_true_class != null || correct_pred != null)) {
			mReprodMedian = MLFuncs.columnMedian(reprod_cc, cvDim);
			
			// Exclude -1's (placeholders) when calculating row means
			if (mPlotPercentAccuracy && correct_pred != null) {
				double[] predPercentAccuracy = MLFuncs.selectedRowMean(correct_pred, -1);
//				boolean debug = true;
//				if (debug) {
//					System.out.println("pred accuracy for each split half: ");
//					NpairsjIO.print(predPercentAccuracy);
//				}
				mPredPercentAccuracyMedian = MLFuncs.median(predPercentAccuracy);
				return true;
			}
			else if (!mPlotPercentAccuracy && pp_true_class != null) {
				double[] ppTrueClassMeans = MLFuncs.selectedRowMean(pp_true_class, -1);
				mPpTrueClassMedian = MLFuncs.median(ppTrueClassMeans);
				return true;
			}
			
		}
		
		return false;
	}
	
	private String removePaddedZeros(String input) {
		String output = input;
		
		while(output.charAt(0) == '0') {
			output = output.substring(1);
		}
		
		return output;
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		if (e.getActionCommand().equals("toggle pc labels") ) {
			boolean showPcLabels = mShowPcLabels.isSelected();
			
			if (showPcLabels) {
				XYItemRenderer renderer = mPlot.getRenderer();
				renderer.setBaseItemLabelsVisible(true);
			} else {
				XYItemRenderer renderer = mPlot.getRenderer();
				renderer.setBaseItemLabelsVisible(false);
			}
		} else if (e.getActionCommand().equals("save values") ) {
			JFileChooser jfc = new JFileChooser();
			jfc.setFileFilter(new FileFilter() {
				public boolean accept(File f) {
			        return f.isDirectory() || f.getName().toLowerCase().endsWith(".csv");
			    }
			
				public String getDescription() {
			        return "Comma Separated Values";
			    }
			});
			int result = jfc.showSaveDialog(this);
			
			if (result == JFileChooser.APPROVE_OPTION) {
				File file = jfc.getSelectedFile();
				
				if(jfc.getFileFilter().getDescription()
						.equals("Comma Separated Values")){
					String saveName = file.getAbsolutePath();
					// trim any trailing "."
					while (saveName.endsWith(".")) {
						saveName = saveName.substring(0, saveName.lastIndexOf("."));
					}
					if (!saveName.endsWith(".csv")) {	
						saveName = saveName.concat(".csv");
					}
					file = new File(saveName);
				}
				
				try {
					BufferedWriter bw = new BufferedWriter(new FileWriter(file) );

					List<XYSeries> seriesList = ((XYSeriesCollection)mPlot.getDataset()).getSeries();
					
					for (XYSeries series : seriesList) {
						bw.write(series.getKey().toString() );
						List<RvpDataItem> items = series.getItems();
						
						for (RvpDataItem item : items) {
							bw.write("," + item.getPcValue() + "pc");
						}
						
						bw.newLine();
						
						bw.write("reproducibility");
						for (RvpDataItem item : items) {
							bw.write("," + item.getXValue() );
						}
						
						bw.newLine();
						
						if (mPlotPercentAccuracy) {
							bw.write("prediction (% acc./100)");
						}
						else {
							bw.write("prediction (post. prob.)");
						}
						for (RvpDataItem item : items) {
							bw.write("," + item.getYValue() );
						}
						
						bw.newLine();
						bw.newLine();
						bw.flush();
					}
										
					bw.close();
				}
				catch (IOException ioex) {
					System.err.println("An error occurred while saving the r and p values to file.");
				}
				
			}
		}
	}
}
