package pls.othertools.rvptool;

import org.jfree.chart.labels.StandardXYItemLabelGenerator;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

@SuppressWarnings("serial")
public class RvpItemLabelGenerator extends StandardXYItemLabelGenerator {
	public String generateLabel(XYDataset dataset, int series, int item) {
		XYSeriesCollection xycoll = (XYSeriesCollection)dataset;
		
		XYSeries xyseries = xycoll.getSeries(series);
		
		RvpDataItem dataitem = (RvpDataItem)xyseries.getItems().get(item);
		
		return dataitem.getPcValue() + "pc";
	}
}
