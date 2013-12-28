/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package pls.chrome.result.view;
import org.jfree.chart.*;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.labels.StandardXYToolTipGenerator;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.urls.StandardXYURLGenerator;
import org.jfree.data.xy.XYDataset;
/**
 * Extends the ChartFactory abstract class so we can create an XYLineChart
 * with vertical lines to differentiate between subjects.
 *
 */
public class EnhancedXY extends ChartFactory{

	public static JFreeChart createSpecialXYLineChart(String title,
			String xAxisLabel,
			String yAxisLabel,
			XYDataset dataset,
			PlotOrientation orientation,
			boolean legend,
			boolean tooltips,
			boolean urls) {

		if (orientation == null) {
			throw new IllegalArgumentException("Null 'orientation' argument.");
		}
		NumberAxis xAxis = new NumberAxis(xAxisLabel);
		xAxis.setAutoRangeIncludesZero(false);
		NumberAxis yAxis = new NumberAxis(yAxisLabel);
		XYItemRenderer renderer = new XYLineAndShapeRenderer(true, false);
		XYPlotChild plot = new XYPlotChild(dataset, xAxis, yAxis, renderer);
		plot.setOrientation(orientation);
		if (tooltips) {
			renderer.setBaseToolTipGenerator(new StandardXYToolTipGenerator());
		}
		if (urls) {
			renderer.setURLGenerator(new StandardXYURLGenerator());
		}

		JFreeChart chart = new JFreeChart(title, JFreeChart.DEFAULT_TITLE_FONT,
				plot, legend);
		ChartTheme currentTheme = getChartTheme();
		currentTheme.apply(chart);
		return chart;

	}
}
