package pls.chrome.result.view;

import org.jfree.chart.labels.StandardXYItemLabelGenerator;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeriesCollection;

public class RegressionItemGenerator extends StandardXYItemLabelGenerator {

  @Override
  public String generateLabel(XYDataset dataset, int series, int item) {
	XYSeriesCollection points = (XYSeriesCollection) dataset;
	RegressionDataItem point = (RegressionDataItem)
			points.getSeries(series).getItems().get(item);

	return point.getSubjectValue();
  }

}
