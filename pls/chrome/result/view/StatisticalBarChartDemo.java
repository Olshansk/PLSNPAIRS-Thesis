package pls.chrome.result.view;

import java.awt.Font;

import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.CategoryAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.renderer.category.CategoryItemRenderer;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;

import extern.AsymmetricStatisticalCategoryDataset;
import extern.AsymmetricStatisticalBarRenderer;

public class StatisticalBarChartDemo extends ApplicationFrame {

    /**
     * Creates a new demo.
     *
     * @param title  the frame title.
     */
    public StatisticalBarChartDemo(final String title) {

        super(title);
        final AsymmetricStatisticalCategoryDataset dataset = createDataset();

        final CategoryAxis xAxis = new CategoryAxis("Type");
        xAxis.setLowerMargin(0.01d); // percentage of space before first bar
        xAxis.setUpperMargin(0.01d); // percentage of space after last bar
        xAxis.setCategoryMargin(0.05d); // percentage of space between categories
        final ValueAxis yAxis = new NumberAxis("Value");

        // define the plot
        final CategoryItemRenderer renderer = new AsymmetricStatisticalBarRenderer();
        final CategoryPlot plot = new CategoryPlot(dataset, xAxis, yAxis, renderer);

        final JFreeChart chart = new JFreeChart("Statistical Bar Chart Demo",
                                          new Font("Helvetica", Font.BOLD, 14),
                                          plot,
                                          true);
        //chart.setBackgroundPaint(Color.white);
        // add the chart to a panel...
        final ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(500, 270));
        setContentPane(chartPanel);

    }
   /**
     * Creates a sample dataset.
     *
     * @return The dataset.
     */
    private AsymmetricStatisticalCategoryDataset createDataset() {

        final AsymmetricStatisticalCategoryDataset result = new AsymmetricStatisticalCategoryDataset();

        result.add(32.5, 17.9, 1.2, "Series 1", "Type 1");
        result.add(27.8, 11.4, 3.2, "Series 1", "Type 2");
        result.add(29.3, 14.4, 3.4, "Series 1", "Type 3");
        result.add(37.9, 10.3, 3.1, "Series 1", "Type 4");

        result.add(22.9,  7.9, 6.6, "Series 2", "Type 1");
        result.add(21.8, 18.4, 8.8, "Series 2", "Type 2");
        result.add(19.3, 12.4, 7.4, "Series 2", "Type 3");
        result.add(30.3, 20.7, 8.8, "Series 2", "Type 4");

        result.add(12.5, 10.9, 6.6, "Series 3", "Type 1");
        result.add(24.8,  7.4, 7.7, "Series 3", "Type 2");
        result.add(19.3, 13.4, 8.8, "Series 3", "Type 3");
        result.add(17.1, 10.6, 1.1, "Series 3", "Type 4");

        return result;

    }

    /**
     * For testing from the command line.
     *
     * @param args  ignored.
     */
    public static void main(final String[] args) {

        final StatisticalBarChartDemo demo = new StatisticalBarChartDemo(
            "Statistical Bar Chart Demo"
        );
        demo.pack();
        RefineryUtilities.centerFrameOnScreen(demo);
        demo.setVisible(true);

    }

}
