package pls.chrome.result.view;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Stroke;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.Plot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.PlotManager;
import pls.chrome.result.model.ResultModel;

/**
 * As far as I can tell, this class is not in use. -Fletcher
 *
 */
public class PredictionRegression extends AbstractPlot implements KeyListener {
	
	PlotManager plotMan = null;
	static boolean plotExists = false;
	static int plotIndex = -1;
	double[] x = null;
	boolean meanOrMedian = true;
	double[] y = null;
	
	public PredictionRegression(GeneralRepository repository, double[] xData,
			boolean bMeanOrMedian, double[] yData) throws IllegalArgumentException { //mean = t; median = f
		//super(repository, "Prediction Regression Plot");
		super("Prediction Regression Plot", repository, "");
		
		x = xData;
		meanOrMedian = bMeanOrMedian;
		y = yData;
		
		if (x != null && y != null) {
			if (x.length != y.length) {
				throw new IllegalArgumentException("X and y arrays must be of the same size (size = number of subjects), and they are not.");
			}
			
			remove(mSubPane);
			
			this.plotMan = repository.getPlotManager();
			addToPlots();
		}
	}

	@Override
	public double[][] getRangeData(ResultModel model) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void makeChart(int fileIndex) {
		if (x != null && y != null)
			mChartPanel.setChart(createChart());
		
	}

	@Override
	public void keyPressed(KeyEvent arg0) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void keyReleased(KeyEvent arg0) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void keyTyped(KeyEvent arg0) {
		// TODO Auto-generated method stub
		
	}
	
	public void addToPlots() {
		if (plotExists && plotIndex != -1 && plotIndex < plotMan.mPlots.size()) {
			//plotExists = true;
			plotMan.mPlots.remove(plotIndex);
			plotMan.mAttachedPlots.remove(true);
		}		
		plotMan.mPlots.add(this);
		plotMan.mAttachedPlots.add(true);
		plotMan.refreshPlots();
		plotIndex = plotMan.mPlots.indexOf(this);
		plotExists = true;
	}

	
	private JFreeChart createChart() {
		XYSeries series = new XYSeries("Point");

		
		for (int i = 0; i < x.length; ++i) {
			//if (x[i] != -1 && y[i] != -1) {
				series.add(x[i], y[i]);
				//System.out.println(x[i] + "," + y[i]);
			//}
		}
		
		XYSeriesCollection mySeries = new XYSeriesCollection();
		mySeries.addSeries(series);
		
		String m = "";
		if (meanOrMedian) m = "Means";
		else m = "Medians";
		
		JFreeChart regressionPlot = ChartFactory.createXYLineChart("Regression Against Prediction Values",
				m, "Loaded Data ",
				mySeries, PlotOrientation.VERTICAL, false, true, false);
		
		XYItemRenderer rend = regressionPlot.getXYPlot().getRenderer();
		Stroke s = new BasicStroke(2);
		rend.setStroke(s);
		
		regressionPlot.getXYPlot().setRenderer(rend);
		
		return regressionPlot;
	}

}
