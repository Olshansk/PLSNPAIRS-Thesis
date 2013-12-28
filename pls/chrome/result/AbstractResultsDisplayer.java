package pls.chrome.result;

import java.io.File;
import java.util.ArrayList;

import javax.swing.JTabbedPane;

import pls.chrome.result.controller.ResultsCommandManager;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.PlotManager;
import pls.chrome.result.model.ResultModel;
import pls.chrome.shared.ProgressDialogWatcher;

/**
 * As far as I can tell, this class is not in use. -Fletcher
 *
 */
public abstract class AbstractResultsDisplayer extends ProgressDialogWatcher {
	protected DetachableTabbedPane mTabbedPane = null;
	protected String mFileName = null;
	protected String mFileDir = null;
	
	protected ResultModel mResultModel = null;
	protected GeneralRepository mRepository;
	
	public AbstractResultsDisplayer(DetachableTabbedPane tabs, String fileName) {
		mTabbedPane = tabs;
		mFileName = fileName;
		mFileDir = new File(fileName).getParent();
	}
	
	public final void doTask() throws Exception {
		mRepository = new GeneralRepository();
		
		PlotManager plotManager = new PlotManager(mTabbedPane, mRepository, progress);
		mRepository.setPlotManager(plotManager);
		 
//		progress.setAlwaysOnTop(true);

//		progress.appendMessage("Loading data ...");
		createResultModel();
//		progress.updateStatus(1);
		
		
		ArrayList<String> viewedResultFiles = new ArrayList<String>();
		viewedResultFiles.add(mFileName);
		
		String type = mRepository.getGeneral().getSelectedDataType();
		mRepository.getGeneral().getViewModel().getViewedLvs(type).add(1);

		mRepository.calculateColourScale();
//		mRepository.setGlobalColourScale(mRepository.getCalculatedColourScale() );
		
//		CommandManager.mResultModel = mResultModel;
		ResultsCommandManager.setRepository(mRepository);
		
		loadCommonPlots();
		
		loadOtherPlots();
		
//		unusedPlots();
		mRepository.getPlotManager().init();

		progress.complete();
	}
	
	protected abstract void createResultModel();
	
	private void loadCommonPlots() {
		//String variableType = mRepository.getGeneral().getVariableType();
		
		// Create brain latent variable plot
//		progress.appendMessage("Creating Brain " + variableType + "s ...");
		
//		mTabbedPane.addTab("Brain " + variableType + "s",
//				new BrainLatentVariablesPlot(mTabbedPane, "Brain " + variableType + "s", getDataTypes(), mRepository));
//		progress.updateStatus(1);
	}
	
	protected abstract void loadOtherPlots();
	
	protected abstract ArrayList<String> getDataTypes();
	
	public GeneralRepository getRepository() {
		return mRepository;
	}
}
