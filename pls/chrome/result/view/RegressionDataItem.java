package pls.chrome.result.view;

import org.jfree.data.xy.XYDataItem;

public class RegressionDataItem extends XYDataItem{

  int subj;
  String splitType;
  
  RegressionDataItem(double x, double y, int subj, String splitType) {
	super(x, y);
	this.subj = subj;
	this.splitType = splitType;
  }

  public String getSubjectValue(){
	return splitType + " "+ subj;
  }

}
