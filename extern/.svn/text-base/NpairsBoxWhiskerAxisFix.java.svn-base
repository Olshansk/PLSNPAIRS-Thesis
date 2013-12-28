package extern;

import java.util.Arrays;
import java.util.List;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.data.statistics.DefaultBoxAndWhiskerCategoryDataset;

/**
 * Fixes the axis on box and whisker plots so drawn outliers are displayed
 * properly. Use in conjunction with  {@link NpairsBoxAndWhiskerRenderer} 
 */
public class NpairsBoxWhiskerAxisFix{

	/**
	 * Our box and whisker plots have been set up to plot 'far out values'.
	 * Values that JFreeChart does not even consider outliers because they are
	 * so annomalous. Currently, even though they are being drawn the axis of
	 * the plot is not correct so they not displayed. Fix the axis while giving
	 * 5% space at the top of the chart.
	 * @param plot the box and whisker plot to fix.
	 */
	public static void fixBoxAndWhiskerRange(CategoryPlot plot) {
		DefaultBoxAndWhiskerCategoryDataset dataset;
		ValueAxis rangeAxis = plot.getRangeAxis();

		double max = Double.NEGATIVE_INFINITY;
		double min = Double.POSITIVE_INFINITY;
		double maxRegular;
		double minRegular;
		double extra;

		dataset = (DefaultBoxAndWhiskerCategoryDataset) plot.getDataset();
		
		//Calculate the max and min over all outliers _including_ 'far outs'.
		//getOutliers() actually returns 'far out' values as well.
		for (int c = 0; c < dataset.getColumnCount(); c++) {
			List<Double> outliers = dataset.getOutliers(0, c);

			for (double out : outliers) {
				max = Math.max(max, out);
				min = Math.min(min, out);
			}
		}

		/* max/min regular represent the max and min values of the plot
		 * (excluding 'far outs'). 'Far outs' are outliers that JFreeChart
		 * considers so far out from any box that they cannot even be considered
		 * outliers and are usually not plotted (except in our case because the
		 * code has been hacked to do so). If all our outliers are below all
		 * the boxes then we want the top of our chart to be the top most point
		 * of any box. The reverse condition applies when all our points are at
		 * the top of all boxes*/

		maxRegular = dataset.getRangeUpperBound(true);
		minRegular = dataset.getRangeLowerBound(true);
		//case #1. All outliers are below upper quartile (q3)
		if (maxRegular > max) {
			max = maxRegular;
		}

		//case #2. All outliers are above lower quartile (q1)
		if (min > minRegular) {
			min = minRegular;
		}

		extra = (max - min) * .05;
		rangeAxis.setUpperBound(max + extra);
		rangeAxis.setLowerBound(min - extra);
	}

	/**
	 * Generate a test dataset so we can be sure that outliers are being
	 * plotted. 
	 * @return mock data.
	 */
	public static DefaultBoxAndWhiskerCategoryDataset mockdatum(){
		DefaultBoxAndWhiskerCategoryDataset dset;
		dset = new DefaultBoxAndWhiskerCategoryDataset();

		Double[] v1 = {3.0,5.3,2.6,7.5,8.6,2000.0};
		Double[] v2 = {5.0,2.0,6.7,8.9};
		Double[] v3 = {4.0,7.8,2.0,8.4,-800.0};

		List<Double> l1 = Arrays.asList(v1);
		List<Double> l2 = Arrays.asList(v2);
		List<Double> l3 = Arrays.asList(v3);

		dset.add(l1,0,0);
		dset.add(l2,1,0);
		dset.add(l3,2,0);
		return dset;
	}
}


