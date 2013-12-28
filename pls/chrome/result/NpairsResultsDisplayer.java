package pls.chrome.result;

import java.util.ArrayList;

import javax.swing.JTabbedPane;

import pls.chrome.result.model.BrainData;
import pls.chrome.result.model.NPairsResultModel;
import pls.chrome.result.view.NpairsCvScoresPlot;
import pls.chrome.result.view.ObservedEigenvaluePlot;
import pls.chrome.result.view.ReproducibilityPlot;

/**
 * This class as far as I can tell is not in use. -Fletcher
 */
public class NpairsResultsDisplayer extends AbstractResultsDisplayer {
	
	private NPairsResultModel mNPairsResultModel = null;
	
	public NpairsResultsDisplayer(DetachableTabbedPane tabs, String fileName) {
		super(tabs, fileName);
	}

	protected void loadOtherPlots() {
		// Plot CV scores
//		progress.appendMessage("Creating CV Scores Plot ...");
//		mTabbedPane.addTab("CV Scores", new NpairsCvScoresPlot(mTabbedPane, "CV Scores", mRepository));
//		progress.updateStatus(1);
		
//		if (mResultModel.getS() != null) {
//			
//			// Plot observed eigenvalues
//			progress.appendMessage("Creating Observed Eigenvalue Plot ...");
//			mTabbedPane.addTab("Observed Eigenvalue Plot", new ObservedEigenvaluePlot(mTabbedPane, "Observed Eigenvalue Plot", mRepository));
//			progress.updateStatus(1);
//		}
		
//		if (mNPairsResultModel.getReprodCC() != null) {
//			
//			// Plot reproducibility
//			progress.appendMessage("Creating Reproducibility Plot ...");
//			mTabbedPane.addTab("Reproducibility", new ReproducibilityPlot(mTabbedPane, "Reproducibility", mRepository));
//			progress.updateStatus(1);
//		}
	}
	
	protected ArrayList<String> getDataTypes() {
		ArrayList<String> dataTypes = new ArrayList<String>();
		dataTypes.add(BrainData.AVG_CANONICAL_STRING);
		dataTypes.add(BrainData.AVG_ZSCORED_STRING);
		dataTypes.add(BrainData.FULL_DATA_STRING);
		
		return dataTypes;
	}

	protected void createResultModel() {
		// Load the model
		NpairsResultLoader loader = new NpairsResultLoader(mFileName);
		
		try {
			loader.loadFile();
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		
		mNPairsResultModel = loader.getNpairsResultModel();
		mResultModel = loader.getResultModel();
		
		// Put the model into the repository
		mRepository.addModel(mFileName, mNPairsResultModel);
		mRepository.getControlPanelModel().initModel();
		mRepository.getPlotManager().refreshPlots();
	}

}