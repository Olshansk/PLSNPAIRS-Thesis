package pls.chrome.result.model;

import java.awt.Component;
import java.util.ArrayList;

import pls.chrome.result.DetachableTabbedPane;
import pls.chrome.result.blvplot.BrainLatentVariablesPlot;
import pls.chrome.result.blvplot.PlotTypeTabbedPane;
import pls.chrome.result.view.*;
import pls.shared.StreamedProgressHelper;


/**
 * A class which creates and destroys plots based on whether or not there are
 * any files with data that applies to that plot.
 */
public class PlotManager {
	private GeneralRepository mRepository;
	private DetachableTabbedPane mTabbedPane;
	public ArrayList<AttachDetachOption> mPlots = new ArrayList<AttachDetachOption>();
	public ArrayList<Boolean> mAttachedPlots = new ArrayList<Boolean>();
	private StreamedProgressHelper mProgress;
	private RegressionPlot rP;

	public PlotManager(DetachableTabbedPane tabbedPane, GeneralRepository repository, StreamedProgressHelper progress)
	{
		mTabbedPane = tabbedPane;
		mRepository = repository;
		mProgress = progress;
		
		mTabbedPane.setPlotManager(this);

	}
	
	public void init() {
		
		//Main brain viewer.
		mProgress.startTask("Initializing main brain viewer");
		mPlots.add(new BrainLatentVariablesPlot("Main Brain Viewer", mRepository) );
		mAttachedPlots.add(true);
		mProgress.endTask();
		
		// PLS Plots
		mProgress.startTask("Initializing PLS plots");
		mPlots.add(new BrainDesignScoresPlot("Brain vs Design Scores", mRepository) );
		mAttachedPlots.add(true);
		mPlots.add(new BrainBehavScoresPlot("Brain vs Behav Scores", mRepository) );
		mAttachedPlots.add(true);
		mPlots.add(new DesignLatentVariablesPlot("Design Latent Variables", mRepository) );
		mAttachedPlots.add(true);
		mPlots.add(new PermutedSingularValuesPlot("Permuted Singular Values", mRepository) );
		mAttachedPlots.add(true);
		mPlots.add(new ResultContrastPanel("Contrasts Information", mRepository) ); 
		mAttachedPlots.add(true);
		mPlots.add(new ObservedSingularValuesPlot("Observed Singular Values Plot", mRepository) );
		mAttachedPlots.add(true);
		mPlots.add(new PercentCrossblockCovariancePlot("Percent Crossblock Covariance", mRepository) );
		mAttachedPlots.add(true);
		mPlots.add(new TaskPLSwithCIPlot("Task PLS with CI Plot", mRepository) );
		mAttachedPlots.add(true);
		mPlots.add(new ResponseFunctionPlot("Response Function Plot", mRepository) ); 
		mAttachedPlots.add(true);
		//mPlots.add(new VoxelIntensityResponsePlot("Voxel Intensity Response Plot", mRepository) );
		//mAttachedPlots.add(true);
		mPlots.add(new TemporalBrainScoresPlot("Temporal Brain Scores Plot", mRepository) ); 
		mAttachedPlots.add(true);
		mPlots.add(new TemporalBrainCorrelationPlot("Temporal Brain Correlation Plot", mRepository) );
		mAttachedPlots.add(true);
		//mPlots.add(new DesignScoresPlot("Design Scores", mRepository) );
		//mAttachedPlots.add(true);
		mProgress.endTask();

		// NPAIRS Plots
		mProgress.startTask("Initializing NPAIRS plots");
		mPlots.add(new NpairsCvScoresPlot("CV Scores", mRepository) );
		mAttachedPlots.add(true);
		mPlots.add(new ObservedEigenvaluePlot("Observed Eigenvalue Plot", mRepository) );
		mAttachedPlots.add(true);
		mPlots.add(new ReproducibilityPlot("Reproducibility", mRepository) );
		mAttachedPlots.add(true);
		mPlots.add(new ReproducibilityPredictionPlot("Reproducibility vs Prediction", mRepository) );
		mAttachedPlots.add(true);


		rP = new RegressionPlot("Regression Plot", mRepository);
		mPlots.add(rP);
		mAttachedPlots.add(false);

		PredictionPlot pP = new PredictionPlot("Prediction Plot", mRepository,rP);
		mPlots.add(pP);
		mAttachedPlots.add(true);

		mProgress.endTask();
		
		mProgress.startTask("Populating plots");
		refreshPlots();
		mProgress.endTask();
		
		// No longer need this item
		mProgress = null;
	}

	public void refreshPlots() {
		Component selectedComponent = mTabbedPane.getSelectedComponent();
		
		mTabbedPane.removeAll();
		
		int selectedIndex = 0;
		int c = 0;
		for (AttachDetachOption plot : mPlots) {
			
			if (mProgress != null) {
				mProgress.startTask(plot.getTitle() );
			}
			
			// Tells the plot to scan the loaded result files and store
			// of the ones that are useful.
			plot.initialize();
			
			// If that plot has no useful result files, don't add it
			// back into the tabs
			if (plot.mResultFilePaths.size() > 0) {
				if (mAttachedPlots.get(c) ) { // attached and has files.
					mTabbedPane.add(plot.getTitle(), plot);
					
					if (plot == selectedComponent) {
						selectedIndex = c;
					}
				}
				else{ //detached and has files.
					if(!(plot instanceof RegressionPlot))
						plot.mDetachedFrame.setVisible(true);
				}
			}
			else if (!mAttachedPlots.get(c) ) { // no files and not attached.
				plot.mDetachedFrame.setVisible(false);
			}
			++c;

			if (mProgress != null) {
				mProgress.endTask();
			}
		}
		
		if (mTabbedPane.getTabCount() > 0) {
			mTabbedPane.setSelectedIndex(selectedIndex);
		}
	}
	
	public void fastRefresh() {
		mTabbedPane.removeAll();
		
		int c = 0;
		for (AttachDetachOption plot : mPlots) {
			// If that plot has no useful result files, don't add it
			// back into the tabs
			if (plot.mResultFilePaths.size() > 0) {
				if (mAttachedPlots.get(c) ) {
					mTabbedPane.add(plot.getTitle(), plot);
				}
				else{
					//we want the regressionplot to stay detached and invisible.
					//if we have closed it.
					if(!(plot instanceof RegressionPlot))
						plot.mDetachedFrame.setVisible(true);
				}
			}
			else if (!mAttachedPlots.get(c) ) {
				plot.mDetachedFrame.setVisible(false);
			}
			++c;
		}
	}
	
	public void attachPlot(AttachDetachOption plot) {
		int i = mPlots.indexOf(plot);
		
		mAttachedPlots.set(i, true);
		fastRefresh();
		mTabbedPane.setSelectedComponent(plot);
	}
	
	public void detachPlot(int index) {
		AttachDetachOption plot = mPlots.get(index);
		plot.detachWindow();
		mAttachedPlots.set(index, false);
		
		fastRefresh();
	}
	
	public void detachPlot(AttachDetachOption plot) {
		int i = mPlots.indexOf(plot);
		
		plot.detachWindow();
		mAttachedPlots.set(i, false);
		fastRefresh(); //fix tabs now that this plot is gone.
	}
	
	public void dispose() {
//		rP.tidyUp();
		mTabbedPane.removeAll();
		mPlots.clear();
	}
	
	public ArrayList<AttachDetachOption> getPlots() {
		return mPlots;
	}
	
	public ArrayList<Boolean> getAttachedPlots() {
		return mAttachedPlots;
	}

}
