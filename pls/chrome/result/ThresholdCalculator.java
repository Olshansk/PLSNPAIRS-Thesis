package pls.chrome.result;

import extern.ArrayFuncs;
import java.util.ArrayList;
import org.apache.commons.math.stat.descriptive.rank.Percentile;

public class ThresholdCalculator {

	/**
	 * Calculates the global threshold by finding the 95th percentile value
	 * of brain value's for all currently open lvs. This method should be called
	 * only when we are examining lvs that belong to either bootstrap and/or
	 * average z-score.
	 * @param data The combined set of brain values of all open lvs.
	 * @return The global threshold, 0 if no lvs are currently viewed, 95th
	 * percentile value otherwise.
	 */
	public static double calculateBootstrapThreshold(ArrayList<Double> data) {
		double weightedThreshold = 0; 
		
		int size = data.size();
		
		if (size > 0) {
			Percentile per = new Percentile();
			double[] values = ArrayFuncs.convertDAL(data);

			weightedThreshold = per.evaluate(values,95);

		}
		
		return weightedThreshold;
	}

	/**
	 * Calculates the global threshold by finding the value 1/3 from the mean
	 * of the max and min. This function should only be used when we are
	 * examining lvs that belong to brainlv, or canonical eigenimages.
	 * @param max 0 if no lvs are being viewed. Otherwise the max value observed
	 * out of all viewed lvs.
	 * @param min 0 if no lvs are being viewed. Otherwise the min value observed
	 * out of all viewed lvs.
	 * @return 0.0 if abs(max) < calculated threshold | abs(min) < calc thresh.
	 * otherwise, ((abs(max) + abs(min))/2)/3
	 */
	public static double calculateNormalThreshold(double max, double min) {
		double threshold = (Math.abs(max) + Math.abs(min)) / 6.0;
		if(Math.abs(max) < threshold || Math.abs(min) < threshold) {
			threshold = 0.0;
		}
		
		return threshold;
	}
}
