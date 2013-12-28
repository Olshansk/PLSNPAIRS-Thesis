package pls.chrome.result;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import javax.swing.JOptionPane;

import pls.chrome.result.controller.ResultsCommandManager;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.NPairsResultModel;
import pls.chrome.result.model.PlotManager;
import pls.chrome.result.model.PlsResultModel;
import pls.chrome.result.model.ResultModel;
import pls.chrome.shared.ProgressDialogWatcher;

public class ResultsDisplayer extends ProgressDialogWatcher {
	private DetachableTabbedPane mTabs;
	//private String mFilename;
	private List<String> mFilenames;
	
	public GeneralRepository mRepository;
	public PlotManager mPlotManager;
	
	public ResultsDisplayer(DetachableTabbedPane tabs, List<String> filenames) {
		mTabs = tabs;
		mFilenames = filenames;
	}
	

	@Override
	public void doTask() throws Exception {
		ResultLoader loader;
		ResultModel model;
		
		int[] dims = null;
		int warningCount = 0;
		String warning="";
				
		//Set up a new repository.
		mRepository = new GeneralRepository();
		ResultsCommandManager.setRepository(mRepository);
		
		//Create the plot manager 
		mPlotManager = new PlotManager(mTabs, mRepository, progress);
		mRepository.setPlotManager(mPlotManager);
		
		for(String file : mFilenames){
			
			//Load the file
			progress.startTask("Loading data: " + file);
			loader = ResultLoader.makeLoader(file);
			loader.loadFile();
			model = loader.getResultModel();
			
			//if the dimensions of this loaded result file are not the same as
			//the previous file do not load it.
			if(dims != null){
				if(!Arrays.equals(dims, model.getDimensions())){
					warning += "The dimensions of " + file + 
					" are not the same as in previously loaded files.\n"; 
					warningCount++;
					progress.endTask();
					continue;
				}
			}else{
				dims = model.getDimensions();
			}
			
			if (model instanceof PlsResultModel) {
				mRepository.addModel(file, (PlsResultModel)model);
			}
			else if (model instanceof NPairsResultModel) {
				mRepository.addModel(file, (NPairsResultModel)model);
			}
			
			String type = mRepository.getGeneral().getSelectedDataType();
			mRepository.getGeneral().getViewModel().getViewedLvs(type).add(1);
			progress.endTask();
		}
		
		// Do some initialization (can be done after models have been loaded)
		mRepository.calculateColourScale();
		mRepository.getControlPanelModel().initModel();
		
		mPlotManager.init();
		progress.complete();
		
		//warn the user if any result files could not be loaded.
		if(!warning.equals("")){
			warning += (warningCount > 0) ? "The result file " : "These result files ";
			warning += "will not be loaded\n";
			JOptionPane.showMessageDialog(mTabs,warning); 
		}
	}
	/*@Override
	public void doTask() throws Exception {
		// Load the data
		progress.startTask("Loading data: " + mFilename);
		ResultLoader loader = ResultLoader.makeLoader(mFilename);
		loader.loadFile();
		ResultModel model = loader.getResultModel();
		progress.endTask();
		
		mRepository = new GeneralRepository();
		
		ResultsCommandManager.mRepository = mRepository;
		
		// Create the plot manager
		mPlotManager = new PlotManager(mTabs, mRepository, progress);
		mRepository.setPlotManager(mPlotManager);
		
		// Add the loaded model to the repository
		if (model instanceof PlsResultModel) {
			mRepository.addModel(mFilename, (PlsResultModel)model);
		}
		else if (model instanceof NPairsResultModel) {
			mRepository.addModel(mFilename, (NPairsResultModel)model);
		}
		
		// Make one of the LV's visible
		String type = mRepository.getGeneral().getSelectedDataType();
		mRepository.getGeneral().getViewModel().getViewedLvs(type).add(1);
		
		// Do some initialization
		mRepository.calculateColourScale();
		mRepository.getControlPanelModel().initModel();
		
//		progress.updateStatus(1);
		
		mPlotManager.refreshPlots();
		
		mPlotManager.init();

		progress.complete();
	}*/
}
