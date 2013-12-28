package pls.chrome.result.view;

import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;

import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.CategoryAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.labels.BoxAndWhiskerToolTipGenerator;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.data.Range;
import org.jfree.data.statistics.DefaultBoxAndWhiskerCategoryDataset;

import pls.shared.GlobalVariablesFunctions;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.NPairsResultModel;
import pls.chrome.result.model.ResultModel;
import pls.shared.MLFuncs;

import extern.NpairsBoxAndWhiskerRenderer;
import extern.NpairsBoxWhiskerAxisFix;
import java.awt.BasicStroke;
import java.awt.Color;

@SuppressWarnings("serial")
public class ReproducibilityPlot extends AbstractPlot implements KeyListener {
	
	public static final int ENTER_KEY = 10;
	
	// The regular expressions used to for parsing the input fields related
	// to CV dim value selection.
	public static final String NUMBER_REGEX = "\\d+";
	public static final String HYPHEN_RANGE_REGEX = "\\d+\\s*-\\s*\\d+";
	public static final String COMMA_RANGE_REGEX = "(\\d+\\s*,\\s*)+\\d+";
	public static final String MULTI_RANGE_REGEX = "(\\d+|\\d+\\s*-\\s*\\d+)(\\s*,\\s*(\\d+|\\d+\\s*-\\s*\\d+))+";
	
	private String mCurrentFileName;
	
	private double[][] mReprodCC;
	
	private HashMap<String, Integer[]> mRangeValues;
	private HashMap<String, String> mRange;
	private int numCvDims;
	
	private JTextField cvDimsField;
	
	private JButton plotButton;
	private JButton resetButton;
	
	private CategoryPlot categoryPlot;
	
	public ReproducibilityPlot(String title, GeneralRepository repository) {
		super(title, repository, GlobalVariablesFunctions.NPAIRS);
		
		mButtonPanel.remove(mLocalButton);
		
		// If one of the variables is null, that means none of the
		// variables have been declared at this point.
		if (cvDimsField == null) {
			cvDimsField = new JTextField(15);
			mRange = new HashMap<String, String>();
			mRangeValues = new HashMap<String, Integer[]>();
		}
		
		JPanel instructionsPane = new JPanel();
		JLabel cvDimsLabel = new JLabel("Select CV dimensions by manual input:");
		instructionsPane.add(cvDimsLabel);
		instructionsPane.add(cvDimsField);
		
		plotButton = new JButton("PLOT");
		resetButton = new JButton("RESET");
		instructionsPane.add(plotButton);
		instructionsPane.add(resetButton);
		
		mSubPane.setLayout(new GridLayout(3, 1, 0, 0));
		mSubPane.add(instructionsPane);
		
		plotButton.addActionListener(this);
		resetButton.addActionListener(this);
		cvDimsField.addKeyListener(this);
	}
	
	public double[][] getRangeData(ResultModel model) {
		return ((NPairsResultModel) model).getReprodCC();
	}
	
	public void makeChart(int fileIndex) {
		NPairsResultModel model = mRepository.getNpairsModel(
				mResultFilePaths.get(fileIndex) );
		mReprodCC = mRangeData.get(fileIndex);
		
		String abbrVariableType = model.getAbbrVariableType().toUpperCase();
		
		mCurrentFileName = model.getFilename();
		numCvDims = mReprodCC[0].length;
		
		// If one of the variables is null, that means none of the
		// variables have been declared at this point.
		if (cvDimsField == null) {
			cvDimsField = new JTextField(15);
			mRange = new HashMap<String, String>();
			mRangeValues = new HashMap<String, Integer[]>();
		}
		
		String range;
		Integer[] rangeValues ;
		if (!mRange.containsKey(mCurrentFileName)) {
			range = "1-" + numCvDims;
			rangeValues = MLFuncs.createIntegerArray(1, numCvDims);
			
			mRange.put(mCurrentFileName, range);
			mRangeValues.put(mCurrentFileName, rangeValues);
		} else {
			range = mRange.get(mCurrentFileName);
			rangeValues = mRangeValues.get(mCurrentFileName);
		}
		
		cvDimsField.setText(range);
		
		// Create our own box and whisker plot
		DefaultBoxAndWhiskerCategoryDataset classDataset = getDataset(rangeValues);
		CategoryAxis categoryAxis = new CategoryAxis(abbrVariableType + " Dimension");
		NumberAxis valueAxis = new NumberAxis("Correlation Coefficient");
		//valueAxis.setAutoRange(false);
//		valueAxis.setDefaultAutoRange(new Range(0.0, 1.0) );
//		valueAxis.setFixedAutoRange(1.0);
		valueAxis.setAutoRangeIncludesZero(false);
		
		
		NpairsBoxAndWhiskerRenderer renderer = new NpairsBoxAndWhiskerRenderer();
		renderer.setBaseToolTipGenerator(new BoxAndWhiskerToolTipGenerator());
		
		categoryPlot = new CategoryPlot(classDataset, categoryAxis, valueAxis, renderer);
		categoryPlot.setRangeZeroBaselineVisible(true);
		categoryPlot.setRangeZeroBaselineStroke(new BasicStroke(1));
		categoryPlot.setRangeZeroBaselinePaint(Color.blue);
		JFreeChart reproducibilityPlot = new JFreeChart("Reproducibility", JFreeChart.DEFAULT_TITLE_FONT, categoryPlot, false);
		
		if (mGlobalButton.isSelected()) {
			valueAxis.setRange(new Range(-1.0, 1.0) );
		}
		/*degenerate case where there is only one cv which has only one value.
		 we need to set the range to be something and for a range to exist we
		 need more than one value.*/
		else if(numCvDims == 1 && mReprodCC.length == 1){
			double val = mReprodCC[0][0];
			valueAxis.setRangeAboutValue(val, 1.0);
		}else{
			NpairsBoxWhiskerAxisFix.fixBoxAndWhiskerRange(categoryPlot);
		}
		
		mChartPanel.setChart(reproducibilityPlot);
	}
	
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == plotButton) {
			plotButtonAction();
		} else if (e.getSource() == resetButton) {
			String range = "1-" + numCvDims;
			Integer[] rangeValues = MLFuncs.createIntegerArray(1, numCvDims);
			
			mRange.put(mCurrentFileName, range);
			mRangeValues.put(mCurrentFileName, rangeValues);
			
			categoryPlot.setDataset(getDataset(rangeValues));
			cvDimsField.setText(range);
		} else {
			super.actionPerformed(e);
		}
	}
    
	public void keyPressed(KeyEvent e) {
		
		// We only handle the case where the enter key is pressed.
		if (e.getKeyChar() != ENTER_KEY) {
			return;
		}
		
		if (e.getSource() == cvDimsField) {
			plotButtonAction();
    	}
	}
	
	private void plotButtonAction() {
		String range = cvDimsField.getText().trim();
		
		// Checks if the required field was filled in first.
		if (range.equals("")) {
			GlobalVariablesFunctions.showErrorMessage("No range was given.");
			return;
			
		// Otherwise, checks if valid values were given as the range.
			
		// Checks if the given range is just a number.
		} else if (range.matches(NUMBER_REGEX)) {
			int cvDimValue = Integer.parseInt(range);
			if (!isCvDimValid(cvDimValue, false)) {
				return;
			}
			
		// Checks if the given range is of the form a-b. Spaces are
		// allowed before and after the hyphen (-).
		} else if (range.matches(HYPHEN_RANGE_REGEX)) {
			String[] newRange = range.split("-");
			int first = Integer.parseInt(newRange[0].trim());
			int last = Integer.parseInt(newRange[1].trim());
			if (!areFirstLastCvDimsValid(first, last)) {
				return;
			}
		
		// Checks if the given range is of the form a, b, c. Spaces are
		// allowed before and after the commas.
		} else if (range.matches(COMMA_RANGE_REGEX)) {
			String[] newRange = range.split(",");
			for (int i = 0; i != newRange.length; i++) {
				int cvDimValue = Integer.parseInt(newRange[i].trim());
				if (!isCvDimValid(cvDimValue, true)) {
					return;
				}
			}
		
		// Checks if the given range is a combination of the two cases
		// above.
		} else if (range.matches(MULTI_RANGE_REGEX)) {
			String[] newRange = range.split(",");
			for (int i = 0; i != newRange.length; i++) {
				String rangeItem = newRange[i].trim();
				if (rangeItem.indexOf('-') != -1) {
					String[] miniRange = rangeItem.split("-");
					int first = Integer.parseInt(miniRange[0].trim());
					int last = Integer.parseInt(miniRange[1].trim());
					if (!areFirstLastCvDimsValid(first, last)) {
						return;
					}
				} else {
					int cvDimValue = Integer.parseInt(rangeItem);
					if (!isCvDimValid(cvDimValue, true)) {
						return;
					}
				}
			}
			
		} else {
			GlobalVariablesFunctions.showErrorMessage("Invalid values were given as the CV dimensions range.");
			return;
		}
		
		Integer[] cvDimValues = getCvDimValues(range);
		categoryPlot.setDataset(getDataset(cvDimValues));
		
		mRange.put(mCurrentFileName, range);
		mRangeValues.put(mCurrentFileName, cvDimValues);
	}
	
	// Retrieves the CV dim values based on the given range. It is assumed
	// here that the given range contains only valid CV dim values.
	private Integer[] getCvDimValues(String range) {
		ArrayList<Integer> numbers = new ArrayList<Integer>();
		
		if (range.matches(NUMBER_REGEX)) {
			int cvDimValue = Integer.parseInt(range);
			numbers.add(cvDimValue);
		} else if (range.matches(HYPHEN_RANGE_REGEX)) {
			String[] miniRange = range.split("-");
			int first = Integer.parseInt(miniRange[0].trim());
			int last = Integer.parseInt(miniRange[1].trim());
			for (int i = first; i <= last; i++) {
				numbers.add(i);
			}
		} else if (range.matches(COMMA_RANGE_REGEX)) {
			String[] miniRange = range.split(",");
			for (int i = 0; i != miniRange.length; i++) {
				int cvDimValue = Integer.parseInt(miniRange[i].trim());
				if (!numbers.contains(cvDimValue)) {
					numbers.add(cvDimValue);
				}
			}
		} else if (range.matches(MULTI_RANGE_REGEX)) {
			String[] newRange = range.split(",");
			for (int i = 0; i != newRange.length; i++) {
				String rangeItem = newRange[i].trim();
				if (rangeItem.indexOf('-') != -1) {
					String[] miniRange = rangeItem.split("-");
					int first = Integer.parseInt(miniRange[0].trim());
					int last = Integer.parseInt(miniRange[1].trim());
					for (int j = first; j <= last; j++) {
						if (!numbers.contains(j)) {
							numbers.add(j);
						}
					}
				} else {
					int cvDimValue = Integer.parseInt(rangeItem);
					if (!numbers.contains(cvDimValue)) {
						numbers.add(cvDimValue);
					}
				}
			}
		}
		
		// Sorts the array in ascending order first before returning it.
		Integer[] cvDimValueIntegers = new Integer[numbers.size()];
		cvDimValueIntegers = numbers.toArray(cvDimValueIntegers);
		Arrays.sort(cvDimValueIntegers);
			
		return cvDimValueIntegers;
	}
	
	// Helper method that checks if the given CV dim value is valid based on
	// the number of dimensions.
	private boolean isCvDimValid(int cvDim, boolean multipleDimsInvolved) {
		if ((cvDim < 1 || cvDim > numCvDims) && multipleDimsInvolved) {
			GlobalVariablesFunctions.showErrorMessage("All given CV dim values must be between 1 and " + numCvDims + ".");
		} else if (cvDim < 1 || cvDim > numCvDims) {
			GlobalVariablesFunctions.showErrorMessage("The given CV dim value must be between 1 and " + numCvDims + ".");
		} else {
			return true;
		}
		
		return false;
	}
	
	// Helper method that checks if the given first and last CV dim values
	// are valid based on the total number of dimensions.
	private boolean areFirstLastCvDimsValid(int first, int last) {
		if (first < 1 || first > numCvDims) {
			GlobalVariablesFunctions.showErrorMessage("The first CV dim value must be between 1 and " + numCvDims + ".");
		} else if (last < 1 || last > numCvDims) {
			GlobalVariablesFunctions.showErrorMessage("The last CV dim value must be between 1 and " + numCvDims + ".");
		} else if (last < first) {
			GlobalVariablesFunctions.showErrorMessage("The first CV dim value should not be larger than the last CV dim value.");
		} else {
			return true;
		}
		
		return false;
	}

	private DefaultBoxAndWhiskerCategoryDataset getDataset(Integer[] cvDims) {
		DefaultBoxAndWhiskerCategoryDataset classDataset = new DefaultBoxAndWhiskerCategoryDataset();
		
		for (int i = 0; i < cvDims.length; i++) {
			int cvDimIndex = cvDims[i] - 1;
			
			ArrayList<Double> values = new ArrayList<Double>();
			for (int j = 0; j != mReprodCC.length; j++) {
				values.add(mReprodCC[j][cvDimIndex]);
			}
			classDataset.add(values, 0, cvDims[i]);
		}
		
		return classDataset;
	}
	
	public boolean addWhiteSpaceAtBottom() {
		return true;
	}
	
	public void keyTyped(KeyEvent e) {}
	public void keyReleased(KeyEvent e) {}
	
	protected boolean ModelIsApplicable(ResultModel model) {
		NPairsResultModel npairsModel = (NPairsResultModel)model;
		
		return npairsModel.getReprodCC() != null;
	}
	
}