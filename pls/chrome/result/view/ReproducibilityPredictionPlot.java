package pls.chrome.result.view;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.event.ActionEvent;

import javax.swing.JCheckBox;
import javax.swing.JComboBox;

import org.jfree.chart.JFreeChart;
import org.jfree.chart.LegendItemCollection;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;

import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import pls.shared.GlobalVariablesFunctions;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.NPairsResultModel;
import pls.chrome.result.model.ResultModel;
import pls.shared.MLFuncs;

@SuppressWarnings("serial")
public class ReproducibilityPredictionPlot extends AbstractPlot {
	
	private JComboBox mCvDimsComboBox = new JComboBox();
	private int mNumCvDims = 0;
	private int mCurrentCvDim = 0;
	
	private boolean allPointsEqual = false;
	
	private JCheckBox mShowLegendChkBox = new JCheckBox("Show Legend", true);
	
	public ReproducibilityPredictionPlot(String title, GeneralRepository repository) {
		super(title, repository, GlobalVariablesFunctions.NPAIRS);
		
		mFilePanel.remove(mFileComboBox);
		mFilePanel.add(mCvDimsComboBox);
		mFilePanel.add(mShowLegendChkBox);
		
		mButtonPanel.remove(mLocalButton);
		mGlobalButton.setText("One Plot Range For All CV Dimensions");
		
		mShowLegendChkBox.addActionListener(this);
	}
	
	protected void updateLvComboBoxAndTabs(int fileIndex) {
		
		if (mCvDimsComboBox == null) {
			mCvDimsComboBox = new JComboBox();
		}
		mCvDimsComboBox.removeActionListener(this);
		mCvDimsComboBox.removeAllItems();
		
		for (int i = 0; i < mNumCvDims; i++) {
			mCvDimsComboBox.addItem("Cv Dim #" + (i + 1));
		}
		mCvDimsComboBox.addActionListener(this);
		
		if (mShowLegendChkBox == null) {
			mShowLegendChkBox = new JCheckBox("Show Legend", true);
		}
	}
	
	public void makeChart(int cvDim) {
		
		// Adjusts the given cv dimension value first if it is out of bounds.
		if (cvDim >= mNumCvDims && mCurrentCvDim < mNumCvDims) {
			cvDim = mCurrentCvDim;
		} else if (cvDim >= mNumCvDims) {
			cvDim = mNumCvDims - 1;
		}
		
		XYSeriesCollection mySeries = new XYSeriesCollection();
		XYSeries allPoints = new XYSeries("main", false);
		
		for (int i = 0; i != mRangeData.size(); i++) {
			
			// Retrieves the median of the reproducibility values
			// for the given cv dimension.
			double[][] reprodCC = mDomainData.get(i);
			double reprodMedian = MLFuncs.columnMedian(reprodCC, cvDim);
			
			// Retrieves the median prediction value.
			double[][] ppTrueClass = mRangeData.get(i);
			// Excludes -1's (placeholders) when calculating row means
			double[] ppTrueClassMeans = MLFuncs.selectedRowMean(ppTrueClass, -1);
			double ppTrueClassMedian = MLFuncs.median(ppTrueClassMeans);
			
			// Represents each point as its own plot so that it can
			// be displayed in the legend as an individual item.
			XYSeries singlePoint = new XYSeries((String) mFileComboBox.getItemAt(i), false);
			singlePoint.add(reprodMedian, ppTrueClassMedian);
			
			// Adds the point to the main plot used as the actual plot.
			allPoints.add(reprodMedian, ppTrueClassMedian);
			
			if(mShowLegendChkBox.isSelected())
				mySeries.addSeries(singlePoint);
		}
		mySeries.addSeries(allPoints);
		
		NumberAxis domainAxis = new NumberAxis("Reproducibility for CV Dim " + (cvDim + 1));
		NumberAxis rangeAxis = new NumberAxis("Prediction");

		domainAxis.setAutoRangeIncludesZero(false);
		rangeAxis.setAutoRangeIncludesZero(false);

		allPointsEqual = checkPoints(allPoints);
		if(mNoneButton.isSelected()){
			if(allPointsEqual){

				//All points are equal so place the point where it will be
				//highly visible in the plot.

				domainAxis = new NumberAxis("Reproducibility for CV Dim " + (cvDim + 1));
				rangeAxis = new NumberAxis("Prediction");
				double[][] point = allPoints.toArray();
				double x = point[0][0];
				double y = point[1][0];
				rangeAxis.setRangeAboutValue(y,1.0);
				rangeAxis.setUpperBound(1.0);
				domainAxis.setRangeAboutValue(x, 1.0);
			}
			//when chart is made autorange is set to default.
		}

		XYPlot plot = new XYPlot(mySeries, domainAxis, rangeAxis, new XYLineAndShapeRenderer());
		plot.setRangeZeroBaselineVisible(true);
		plot.setRangeZeroBaselineStroke(new BasicStroke(1));
		plot.setRangeZeroBaselinePaint(Color.blue);

		JFreeChart chart = new JFreeChart("Reproducibility vs Prediction", plot);
		
		// If the legend is being displayed, all items from the existing
		// legend are shown except for the one representing the main plot,
		// which is the last item. This is because we only want items
		// representing each individual point.
		LegendItemCollection newLegend = new LegendItemCollection();
		if (mShowLegendChkBox.isSelected()) {
			LegendItemCollection legend = plot.getLegendItems();
			for (int i = 0; i != legend.getItemCount() - 1; i++) {
				newLegend.add(legend.get(i));
			}
		}
		plot.setFixedLegendItems(newLegend);

		globalRangeMin = 0;
		if (mGlobalButton.isSelected()) {
			/* degenerate case where globalDomainMin == globalDomainMax. This
			 * happens when we have only a single cv and a single value or
			 * all the domain values across all cvs are the same. If this is
			 * true then 'extra' in applyMinMaxValues is 0 and the value is
			 * pressed up against the left hand x-axis
			 */
			if(globalDomainMin == globalDomainMax){
				double[][] point = allPoints.toArray();
				double x = point[0][0];
				ValueAxis domain = plot.getDomainAxis();

				applyMinMaxValues(chart, GLOBAL, RANGE);
				domain.setRangeAboutValue(x, 1.0);
			}else{
				applyMinMaxValues(chart, GLOBAL, RANGE);
				applyMinMaxValues(chart, GLOBAL, DOMAIN);
			}
			/*Another degenerate case, it is possible that all the values
		 	 * have a y value of 0. In this case the point would be squished at
			 * the bottom of the screen */
			if (globalRangeMin == globalRangeMax) {
				ValueAxis range = plot.getRangeAxis();
				range.setRangeAboutValue(0, 1.0);
			}
		}


		mChartPanel.setChart(chart);
		
		mCurrentCvDim = cvDim;
	}
	
	public void initialize() {
		mNumCvDims = Integer.MAX_VALUE;
		
		super.initialize();
	}
	
	public double[][] getRangeData(ResultModel model) {
		NPairsResultModel newModel = (NPairsResultModel) model;
		return newModel.getPPTrueClass();
	}
	
	public double[][] getDomainData(ResultModel model) {
		NPairsResultModel newModel = (NPairsResultModel) model;
		double[][] reprodCC = newModel.getReprodCC();
		
		// Retrieves the minimum number of cv dims from each
		// result file that is loaded.
		mNumCvDims = Math.min(mNumCvDims, reprodCC[0].length);
		
		return reprodCC;
	}
	
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == mCvDimsComboBox || e.getSource() == mGlobalButton 
			|| e.getSource() == mLocalButton || e.getSource() == mNoneButton
			|| e.getSource() == mShowLegendChkBox) {
			
			int cvDim = mCvDimsComboBox.getSelectedIndex();
			makeChart(cvDim);
		} else {
			super.actionPerformed(e);
		}
	}
	
	public boolean addWhiteSpaceAtBottom() {
		return true;
	}
	
	// Returns true if all the points in the given XYSeries are the same.
	// Otherwise, returns false.
	private boolean checkPoints(XYSeries allPoints) {
		boolean result = true;
		
		double[][] array = allPoints.toArray();
		double x = array[0][0];
		double y = array[1][0];
		
		for (int i = 1; i != array[0].length; i++) {
			double currX = array[0][i];
			double currY = array[1][i];
			
			if (currX != x || currY != y) {
				result = false;
				break;
			}
		}
		
		return result;
	}
	
	protected boolean ModelIsApplicable(ResultModel model) {
		NPairsResultModel npairsModel = (NPairsResultModel) model;
		
		return npairsModel.getReprodCC() != null 
			&& npairsModel.getPPTrueClass() != null;
	}
}