package pls.chrome.result.model;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.CategoryAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.labels.StandardCategoryToolTipGenerator;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.category.StackedBarRenderer;
import org.jfree.chart.urls.StandardCategoryURLGenerator;
import org.jfree.data.category.CategoryDataset;


public class ChartFactoryChild extends ChartFactory {

	/**
	  834        * Creates a stacked bar chart with default settings.  The chart object 
	  835        * returned by this method uses a {@link CategoryPlot} instance as the
	  836        * plot, with a {@link CategoryAxis} for the domain axis, a 
	  837        * {@link NumberAxis} as the range axis, and a {@link StackedBarRenderer} 
	  838        * as the renderer.
	  839        *
	  840        * @param title  the chart title (<code>null</code> permitted).
	  841        * @param domainAxisLabel  the label for the category axis 
	  842        *                         (<code>null</code> permitted).
	  843        * @param rangeAxisLabel  the label for the value axis 
	  844        *                        (<code>null</code> permitted).
	  845        * @param dataset  the dataset for the chart (<code>null</code> permitted).
	  846        * @param orientation  the orientation of the chart (horizontal or 
	  847        *                     vertical) (<code>null</code> not permitted).
	  848        * @param legend  a flag specifying whether or not a legend is required.
	  849        * @param tooltips  configure chart to generate tool tips?
	  850        * @param urls  configure chart to generate URLs?
	  851        *
	  852        * @return A stacked bar chart.
	  853        */
public static JFreeChart createStackedBarChart(String title,
	String domainAxisLabel,
	String rangeAxisLabel,
	CategoryDataset dataset,
	PlotOrientation orientation,
	boolean legend,
	boolean tooltips,
	boolean urls) {
	if (orientation == null) {
		throw new IllegalArgumentException("Null 'orientation' argument.");
	}
	CategoryAxis categoryAxis = new CategoryAxis(domainAxisLabel);
	ValueAxis valueAxis = new NumberAxis(rangeAxisLabel);
	
	StackedBarRendererChild renderer = new StackedBarRendererChild();
	if (tooltips) {
		renderer.setBaseToolTipGenerator(
				new StandardCategoryToolTipGenerator());
	}
		if (urls) {
			renderer.setBaseItemURLGenerator(
					new StandardCategoryURLGenerator());
		}

		CategoryPlot plot = new CategoryPlot(dataset, categoryAxis, valueAxis, 
				renderer);
		plot.setOrientation(orientation);
		JFreeChart chart = new JFreeChart(title, JFreeChart.DEFAULT_TITLE_FONT,
				plot, legend);
	  		
		return chart;
	}

}
