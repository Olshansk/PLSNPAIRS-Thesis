package pls.chrome.result.view;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import javax.swing.JButton;
import javax.swing.JFrame;

/**
 * A class for testing the creation of the regression plot with various
 * different datasets.
 */
public class testRegressionPlot {
  int testNum;
  RegressionPlot plot;

  testRegressionPlot(RegressionPlot plot){
	this.plot = plot;
	testNum = 0;

	JFrame testFrame = new JFrame();

	JButton nextTest = new JButton("Next test");

	nextTest.addActionListener(new ActionListener() {

		  public void actionPerformed(ActionEvent e) {
			testNum++;
			testData();
		  }
	});
	testFrame.add(nextTest);
	testFrame.setVisible(true);
	testFrame.setTitle("Regression Plot test");
	testFrame.pack();
	
  }
  /** Use this function to provide mock data for sanity testing
	 *
	 */
	public void testData(){

	  ArrayList<Double> rvals = new ArrayList<Double>();
	  ArrayList<Double> means = new ArrayList<Double>();
	  ArrayList<Double> medians = new ArrayList<Double>();

	  switch(testNum){
		case 1:

		  //test t1
		  rvals.add(14.2); rvals.add(16.4); rvals.add(12.8);
		  rvals.add(13.2); rvals.add(15.8); rvals.add(17.3);

		  means.add(13.5); means.add(5.3); means.add(-1.0);
		  means.add(8.6);  means.add(-1.0);  means.add(9.7);

		  medians.add(5.7); medians.add(7.2); medians.add(3.6);
		  medians.add(-1.0);

		  plot.drawPlot(rvals, means, medians, "test1y", "test1x","session");
		  break;

		case 2:

		  //test2 more x vals than y vals
 		  rvals.add(14.2); rvals.add(-1.0);  rvals.add(-1.0);
		  rvals.add(13.2); rvals.add(15.8);  rvals.add(17.3);

		  means.add(13.5); means.add(5.3); means.add(-1.0);
		  means.add(8.6);  means.add(-1.0); means.add(9.7);
		  means.add(8.9);

		  medians.add(5.7); medians.add(7.2); medians.add(3.6);
		  medians.add(-1.0);

		  plot.drawPlot(rvals, means, medians, "test2y", "test2x","session");
		  break;

		case 3:

		  //test3 more y values than x values
		  rvals.add(14.2);  rvals.add(-1.0); rvals.add(-1.0);
		  rvals.add(13.2);  rvals.add(15.8); rvals.add(17.3);
		  rvals.add(8.9);

		  means.add(13.5);  means.add(5.3);  means.add(-1.0);
		  means.add(8.6);  means.add(-1.0);  means.add(9.7);

		  medians.add(5.7); medians.add(7.2); medians.add(3.6);
		  medians.add(-1.0);

		  plot.drawPlot(rvals, means, medians, "test3y", "test3x","session");
		  break;

		case 4:
		  //test4 a single y value

		  rvals.add(2.3);
		  means.add(13.5);  means.add(5.3);	  means.add(-1.0);
		  means.add(8.6);  means.add(-1.0);	  means.add(9.7);

		  medians.add(5.7); medians.add(7.2); medians.add(3.6);
		  medians.add(-1.0);

		  plot.drawPlot(rvals, means, medians, "test4y", "test4x","session");
		  break;

		case 5:

		  //test5 no y values
		  means.add(13.5);  means.add(5.3);	  means.add(-1.0);
		  means.add(8.6);  means.add(-1.0);	  means.add(9.7);

		  medians.add(5.7); medians.add(7.2); medians.add(3.6);
		  medians.add(-1.0);

		  plot.drawPlot(rvals, means, medians, "test5y", "test5x","session");
		  break;

		case 6:
		  //test6 no x values
		  rvals.add(14.2);  rvals.add(16.4);  rvals.add(12.8);
		  rvals.add(13.2);  rvals.add(15.8);  rvals.add(17.3);

		  plot.drawPlot(rvals, means, medians, "test6y", "test6x","session");
		  break;

		case 7:

		  //test 7 x values all the same
		  rvals.add(14.2);  rvals.add(16.4);  rvals.add(12.8);
		  rvals.add(13.2);  rvals.add(15.8);  rvals.add(17.3);
		  means.add(4.0);  means.add(4.0);	  means.add(4.0);

		  medians.add(5.7); medians.add(7.2); medians.add(3.6);
		  medians.add(-1.0);

		  plot.drawPlot(rvals, means, medians, "test7y", "test7x","session");
		  break;

		case 8:
		  //test 8 mask off the all but the 5th subject
		  rvals.add(-1.0);  rvals.add(-1.0);  rvals.add(-1.0);
		  rvals.add(-1.0);  rvals.add(8.0);	  rvals.add(-1.0);

		  means.add(13.5);  means.add(5.3);	  means.add(-1.0);
		  means.add(8.6);  means.add(6.0);	  means.add(9.7);

		  medians.add(5.7); medians.add(7.2); medians.add(3.6);
		  medians.add(-1.0);

		  plot.drawPlot(rvals, means, medians, "test8y", "test8x","session");
		  break;
	  }
	}

}
