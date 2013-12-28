package pls.chrome.result.blvplot;

import java.util.ArrayList;

import javax.swing.BoxLayout;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSplitPane;

import pls.chrome.result.controller.ResultsCommandManager;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.view.AttachDetachOption;

/**
 * Main plot window. The "Main brain viewer" plot.
 */
@SuppressWarnings("serial")
public class BrainLatentVariablesPlot extends AttachDetachOption
{
	
	public static final int ENTER_KEY = 10;
	
	private PlotTypeTabbedPane mTabbedPane = null;
	
	private LeftSidePanel mLeftSidePanel = null;
	
	private JScrollPane mLeftSideScroller = null;

	private JSplitPane splitPane;

	private GeneralRepository mRepository;

	private int dividerLocation;
	
	public BrainLatentVariablesPlot(String title, GeneralRepository repository)
	{
		super(repository, title);
		
		mRepository = repository; 
		
		setLayout(new BoxLayout(this, BoxLayout.X_AXIS) );
		setAlignmentY(JPanel.TOP_ALIGNMENT);
		
		ResultsCommandManager.mPlot = this;
		
		// Create the left side panel
		mLeftSidePanel = new LeftSidePanel(repository);
		
		// Create the tabbed pane (BrainInfoPane/ThreePaneViewer)
		mTabbedPane = new PlotTypeTabbedPane(repository);

		// Set up the button for attaching/detaching from the main
		// results displayer.
		mLeftSidePanel.add(mAttachDetachButton, 0);
		
		// Create a scroll pane for the left side panel (for when we resize it)
		mLeftSideScroller = new JScrollPane(mLeftSidePanel);
		
		// Create a split pane to allow us to resize the left side panel
		splitPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT,
				mLeftSideScroller,
				mTabbedPane);
		splitPane.setOneTouchExpandable(true);
		splitPane.setResizeWeight(0.33);
		dividerLocation = splitPane.getDividerLocation();

		// Add the split pane
		add(splitPane);
	}

	/**
	 * Sets the left side panel visible or invisible. Remembers the positioning
	 * of the main brain viewer and the left side panel so that when the left
	 * side panel is made visible again the user sees what they saw before the
	 * left side panel was made invisible.
	 * @param isVisible true if the left side panel is invisible. 
	 */
	public void setLeftSidePanelVisible(boolean isVisible){
		if(isVisible){
			splitPane.setDividerLocation(dividerLocation);
			mLeftSideScroller.setVisible(true);
		}else{
			//get last set divider location so we can set it to that location
			//when the left side bar is made visible again.
			dividerLocation = splitPane.getDividerLocation();
			mLeftSideScroller.setVisible(false);
		}
	}

	@Override
	public void initialize() {
		mResultFilePaths = new ArrayList<String>(mRepository.getModels() );
	}
	
	public PlotTypeTabbedPane getPlotTabs(){ return mTabbedPane;}
	
}