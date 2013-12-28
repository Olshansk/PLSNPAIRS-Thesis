package pls.chrome.result.view;

import java.awt.BorderLayout;
import java.awt.event.ActionEvent;
import java.util.ArrayList;

import javax.swing.JCheckBox;
import javax.swing.JComboBox;

import org.jfree.chart.JFreeChart;
import org.jfree.chart.LegendItemCollection;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.StandardXYItemRenderer;
import org.jfree.chart.labels.StandardXYToolTipGenerator;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.chart.axis.NumberAxis;

import pls.shared.GlobalVariablesFunctions;
import pls.chrome.result.DomainAndRangeSetter;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.PlsResultModel;
import pls.chrome.result.model.ResultModel;
import pls.chrome.result.controller.observer.Event;
import pls.chrome.result.controller.observer.datachange.*;
import pls.shared.MLFuncs;

@SuppressWarnings("serial")
public class BrainDesignScoresPlot extends AbstractPlot implements DataChangeObserver{
	
	private JComboBox mLvComboBox = new JComboBox();
	private JCheckBox mShowLegendChkBox = new JCheckBox("Show Legend", true);
	
	private DomainAndRangeSetter mDnrSetter = new DomainAndRangeSetter(0);
	
	public BrainDesignScoresPlot(String title, GeneralRepository repository) {
		super(title, repository, GlobalVariablesFunctions.PLS);

        mRepository.getPublisher().registerObserver(this);
                
		mFilePanel.add(mLvComboBox);
		mFilePanel.add(mShowLegendChkBox);
		mFilePanel.add(mDnrSetter);
		
		mSubPane.remove(mButtonPanel);
		mSubPane.remove(mFilePanel);
		
		mSubPane.setLayout(new BorderLayout());
		mSubPane.add(mButtonPanel, BorderLayout.NORTH);
		mSubPane.add(mFilePanel, BorderLayout.CENTER);
		
		mShowLegendChkBox.addActionListener(this);
	}
	
	public double[][] getRangeData(ResultModel model) {
		return model.getBrainScores();
	}
	
	public double[][] getDomainData(ResultModel model) {
		return model.getDesignScores();
	}
	
	protected void updateLvComboBoxAndTabs(int fileIndex) {
		ResultModel model = mRepository.getGeneral(mResultFilePaths.get(fileIndex) );
		int numLvs = model.getBrainData().getNumLvs();
		String variableType = model.getVariableType();
		
		// Adds a dropdown menu for selecting a different LV.
		if (mLvComboBox == null) {
			mLvComboBox = new JComboBox();
		}
		mLvComboBox.removeActionListener(this);
		mLvComboBox.removeAllItems();
		
		for (int i = 0; i < numLvs; i++) {
			mLvComboBox.addItem(variableType + " #" + (i + 1));
		}
		mLvComboBox.addActionListener(this);
		
		if (mDnrSetter == null) {
			mDnrSetter = new DomainAndRangeSetter(0);
		}
		mDnrSetter.setNumPlots(numLvs);
		
		if (mShowLegendChkBox == null) {
			mShowLegendChkBox = new JCheckBox("Show Legend", true);
		}
	}
	
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == mLvComboBox || e.getSource() == mGlobalButton 
				|| e.getSource() == mLocalButton || e.getSource() == mNoneButton
				|| e.getSource() == mShowLegendChkBox) {
			int file = mFileComboBox.getSelectedIndex();
			int lv = mLvComboBox.getSelectedIndex();
			makeLvChart(file, lv);
		} else {
			super.actionPerformed(e);
		}
	}

	//fileindex represents the selected index value of the jcombobox
	//in the braindesignscores plot.
	public void makeChart(int fileIndex) {
		localRangeMin = Double.POSITIVE_INFINITY;
		localRangeMax = Double.NEGATIVE_INFINITY;
		
		localDomainMin = Double.POSITIVE_INFINITY;
		localDomainMax = Double.NEGATIVE_INFINITY;
		
		// Retrieves the overall min and max values for both the domain data
		// and the range data of the model of the given file index. this is local
		//to the specific file we are examinging.
		double[][] brainScores = mRangeData.get(fileIndex);
		getMinMaxValues(brainScores, LOCAL, RANGE);
		
		double[][] designScores = mDomainData.get(fileIndex);
		getMinMaxValues(designScores, LOCAL, DOMAIN);
		
		// Uses the first LV as the default selected LV.
		makeLvChart(fileIndex, 0);
	}


	/**
	 * Draws the Brain vs Design scores plot. Is called by makeChart().
	 * @param fileIndex the selected index value of the jcombobox
	 * @param c the desired lv we want to plot.
	 */
	public void makeLvChart(int fileIndex, int c) {
		//Get the model associated with the selected file and LV.
		ResultModel model = mRepository.getGeneral(mResultFilePaths.get(fileIndex) );

		//brainScores[values][LV], designScores[values][LV]
		double[][] brainScores = mRangeData.get(fileIndex);
		double[][] designScores = mDomainData.get(fileIndex);

		//ATOL, TOL, VOWEL are examples of conditionNames
		ArrayList<String> conditionNames = model.getConditionNames();

		//type we are dealing with i.e "LV"
		String abbrVariableType = model.getAbbrVariableType().toUpperCase();
		
		int numConditions = conditionNames.size();

		//this variable's length denotes the number of groups
		//index i refers to group i. the actual value means the number of
		//subjects in a group?
		int[] numSubjectList = model.getNumSubjectList(); 

		boolean b = mShowLegendChkBox.isSelected();

		double[] bScr = MLFuncs.getColumn(brainScores, c);
		double[] dScr = MLFuncs.getColumn(designScores, c);

		ArrayList<ArrayList<XYSeries> > someSeries = new ArrayList<ArrayList<XYSeries> >();

		XYSeriesCollection dataset = new XYSeriesCollection();
		StandardXYToolTipGenerator tooltip = new StandardXYToolTipGenerator();
		StandardXYItemRenderer renderer = new StandardXYItemRenderer(
				StandardXYItemRenderer.SHAPES, tooltip);

		//for each group
		for (int i = 0; i < numSubjectList.length; i++) {
                    //for each group create a new XYSeries arraylist
                    //intuitively this means that each group will contain a list of xyseries
                    //each of which coresponds to a plotted condition.
                    someSeries.add(new ArrayList<XYSeries>());
                    int numSubjects = numSubjectList[i];

                    //unfortunatley we need to iterate in this way because our dScr, and bScr
                    //vars are one dimensional. but basically, for each condition in a group and for
                    //each subject create an XYSeries series using the data within {d,b}Scr.
                    for (int j = i * numConditions, conditionNumber = 0; j < (i * numConditions)
                                    + numConditions; j++, conditionNumber++) {

                        for (int k = j * numSubjects; k < (j * numSubjects) + numSubjects; k++) {
                            //as we iterate through the points to plot for each subject check if a xy series
                            //has been set up for that subject. if it has, add data to that series, otherwise
                            //create a new series and add data to it. remember a series is essentially a
                            //(group,conditionNum) tuple.
                            XYSeries series = null;
                            //each 'series' corresponds to an individual condition
                            //if the conditionNumber we are looking at is greater than the current group's
                            //size then this means we have never added plot data for the condition in
                            //question to the current group. Otherwise if it is less, then we have previously
                            //entered data for that condition into the current group we are examining.
                            if (conditionNumber < someSeries.get(i).size() ) {
                                series = someSeries.get(i).get(conditionNumber);
                            }
                            else {
                                //for the current group we are working with we encountered a new condition
                                //that we have not added data for, so we add a new xy series for the condition.
                                series = new XYSeries("Group: " + (i + 1) + ", Condition: " +
                                                conditionNames.get(conditionNumber) );
                                someSeries.get(i).add(series);
                            }
                            series.add(dScr[k], bScr[k]);
                        }
                    }
		}

		for (ArrayList<XYSeries> j : someSeries) {
			for (XYSeries s : j) {
				dataset.addSeries(s);
			}
		}

		NumberAxis domainAxis = new NumberAxis("Design Scores");
		domainAxis.setAutoRangeIncludesZero(false);
		NumberAxis rangeAxis = new NumberAxis("Brain Scores");
		rangeAxis.setAutoRangeIncludesZero(false);

		XYPlot plot = new XYPlot(dataset, domainAxis, rangeAxis, renderer);
		

                JFreeChart chart = new JFreeChart("Brain vs. Design Scores, "
				+ abbrVariableType + " " + (c + 1), plot);
		
		if (!b) {
			plot.setFixedLegendItems(new LegendItemCollection() );
		}
                
		//changes the axis if either of these two buttons are set.
		//if "individual plot ranges" is selected instead, then there is no
		//axis modification.
		if (mGlobalButton.isSelected()) { //"One plot range for all result files"
   			applyMinMaxValues(chart, GLOBAL, RANGE);
			applyMinMaxValues(chart, GLOBAL, DOMAIN);
		} else if (mLocalButton.isSelected()){ //"One plot range per result file"
                        applyMinMaxValues(chart, LOCAL, RANGE);
			applyMinMaxValues(chart, LOCAL, DOMAIN);
		}
                                
		mDnrSetter.setAxes(domainAxis, rangeAxis, c);
		mDnrSetter.updateInputFields(); //updates the PLOT text fields


		mChartPanel.setChart(chart); //update panel.
	}

	@Override
	protected boolean ModelIsApplicable(ResultModel model) {
		PlsResultModel plsModel = (PlsResultModel)model;
		return plsModel.getDesignScores() != null && plsModel.getBrainScores() != null;
	}

        @Override
	public void notify(FlipVolumeEvent e) {}

	@Override
	public void notify(LoadedVolumesEvent e) {}

	@Override
	/** Method that is run when an InvertedLvEvent is triggered.
	 * This happens when the 'invert cv/lv' checkbox is set
	 * @param e the InvertedLvEvent trigger.
	 */
	public void notify(InvertedLvEvent e) {
                initialize();
	}

	@Override
	public void notify(Event e) {}
	
}