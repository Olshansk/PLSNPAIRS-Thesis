package pls.chrome.result.blvplot;

import java.awt.Dimension;

import javax.swing.JTabbedPane;

import pls.chrome.result.ThreeViewPanel;
import pls.chrome.result.clusterreport.ClusterReportPanel;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.PlotManager;
import pls.chrome.result.view.scatterplot.EnhancedScatterPlot;

@SuppressWarnings("serial")
public class PlotTypeTabbedPane extends JTabbedPane
{	
	BrainInfoPane mBrainPane = null;
	ClusterReportPanel mClusterReportPanel = null;
	ThreeViewPanel mThreeViewPanel = null;
	EnhancedScatterPlot mScatterPlotPane = null;

	public PlotTypeTabbedPane(GeneralRepository repository)
	{
		mBrainPane = new BrainInfoPane(repository);
		mClusterReportPanel = new ClusterReportPanel(repository);
		mThreeViewPanel = new ThreeViewPanel(repository);
		mScatterPlotPane = new EnhancedScatterPlot("Scatter Plot", 
													repository, this);
		
		addTab("Single Plane Viewer", mBrainPane);
		addTab("Three Plane Viewer", mThreeViewPanel);
		addTab("Scatter Plot", mScatterPlotPane);
		addTab("Cluster Report", mClusterReportPanel);
		
		// NOTE:  This is necessary in order for this window to get small.
		// For some reason, its minimum width is 680 by default which keeps
		// the vertical scroll bars of our windows off screen once we resize
		// the main window to a width smaller than 680.
		setMinimumSize(new Dimension(100, 100) );
	}
	
}
