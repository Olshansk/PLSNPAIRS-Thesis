package pls.chrome.result;

import java.awt.Dimension;
import java.awt.Frame;
import java.io.IOException;
import java.io.PipedOutputStream;
import java.util.List;

import javax.swing.JFrame;

import pls.chrome.result.blvplot.BrainLatentVariablesPlot;
import pls.chrome.result.blvplot.PlotTypeTabbedPane;
import pls.chrome.result.controller.ResultsCommandManager;
import pls.chrome.result.controller.observer.Event;
import pls.chrome.result.controller.observer.filters.BrainFilterEvent;
import pls.chrome.result.controller.observer.filters.FiltersObserver;
import pls.chrome.result.controller.observer.filters.IncorrectLagsSelectedEvent;
import pls.chrome.result.controller.observer.filters.SliceFiltersEvent;
import pls.chrome.result.controller.observer.filters.ViewedLvsEvent;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.PlotManager;
import pls.shared.StreamedProgressDialog;
import pls.shared.StreamedProgressHelper;

/**
 * The highest level awt component of the results viewer.  Created by the main
 * PLS application when a results file is selected for viewing.
 */
@SuppressWarnings("serial")
public class ResultFrame2 extends JFrame implements FiltersObserver {
	
	public static final Dimension DIMENSION = new Dimension(1280, 960);

	private DetachableTabbedPane mTabs;
	
	//private String mFilename;
	private List<String> mFilenames;
	
	private GeneralRepository mRepository;
	
	private PlotManager mPlotManager;

	public ResultFrame2(String fileType, List<String> fileNames) {
		super(fileType + " Results");
		setDefaultCloseOperation(DISPOSE_ON_CLOSE);

		mFilenames = fileNames;
		
		mTabs = new DetachableTabbedPane();
		add(mTabs);
				
		load();
		
		setSize(DIMENSION); // For gui platforms that can't handle maximize
		setExtendedState(getExtendedState() | Frame.MAXIMIZED_BOTH); // Try to maximize
	}
	
	/**
	 * Creates a ResultsDisplayer object, which loads data from a given file
	 * and creates a repository and plot manager.  ResultsDisplayer is run in
	 * another thread using the ProgressDialog class.
	 */
	private void load() {
		try {
			final StreamedProgressDialog dialog = new StreamedProgressDialog(this, 64);
			PipedOutputStream pos = new PipedOutputStream();
			dialog.connectWriter(pos);
			StreamedProgressHelper helper = new StreamedProgressHelper();
			helper.addStream(pos);
		
			final ResultsDisplayer displayer = new ResultsDisplayer(mTabs, mFilenames);
			
			dialog.worker = displayer;
			
			displayer.progress = helper;
			displayer.start();
			
			// Start a new thread that waits on the displayer.
			// This allows the ResultFrame to be responsive in the mean time.
			new Thread() {
				public void run() {
					// Wait until the dialog is "complete", i.e. the Repository
					// and PlotManager have been loaded.
					while (!dialog.isComplete()) {
						try {
							sleep(250);
						}
						catch (InterruptedException e){
							continue;
						}
					}
					
					mRepository = displayer.mRepository;
					mPlotManager = displayer.mPlotManager;
					
					//Get the main brain viewer window which is always at 
					//position 0 if it is the first plot to be added when 
					//initializing the plot manager and it is set as attached.
//					BrainLatentVariablesPlot mainView;
//					mainView = (BrainLatentVariablesPlot) 
//								mPlotManager.getPlots().get(0);
//					
//					postLoadStuff(mainView.getPlotTabs());
					postLoadStuff();
					
				}
			}.start();
		} catch (IOException ioex) {
			ioex.printStackTrace();
		}
	}
	
	private void postLoadStuff() {
		// Now that we have a repository, create the menu bar
		setJMenuBar(new ResultMenuBar2(mRepository, mTabs) );

		mRepository.getPublisher().registerObserver(this);
		setVisible(true); // Make the frame visible
		repaint();
	}
	
	public void dispose() {
		// Tell the plot manager to unload
		mPlotManager.dispose();
		
		// Tell the repository to destroy itself
		mRepository.dispose();
		
		// Tell the command manager to dispose of its static references
		//(should eventually not be needed)
		ResultsCommandManager.dispose();
		
		super.dispose();
	}
	
	public void finalize() throws Throwable {
		
		super.finalize();
	}

	@Override
	public void notify(SliceFiltersEvent e) {
		// Just adding images to the main brain viewer doesn't seem to
		// cause a repaint on some systems.  This ensures the frame is
		// redrawn whenever we change the slices shown.
		repaint();
	}

	@Override
	public void notify(ViewedLvsEvent e) {}

	@Override
	public void notify(Event e) {}

	@Override
	public void notify(BrainFilterEvent e) {
		// TODO Auto-generated method stub
		
	}
	public void notify(IncorrectLagsSelectedEvent e){}

}