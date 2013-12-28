package pls.shared;

import org.apache.commons.math.MathException;
import org.apache.commons.math.special.Erf;


public class PValue {
	public static double ThresholdToPValue(double threshold, double mu, double sigma) {
		
		double pvalue = 0.0;
		try {
			pvalue =  (1 + Erf.erf( (threshold - mu) / (Math.sqrt(2) * sigma))) / 2;
			pvalue = (1 - pvalue) * 2;
		}
		catch (MathException me) {
			//As the threshold grows in size the p value approaches zero.
			//This exception happens when the threshold is simply too large 
			//(values > 50) so we can simply return 0; A threshold value of 37
			//returns a p value of 1.0658 x 10^-14
			//TODO: Figure out definitively what is happening here. Apparently 
			//having a large (in magnitude) negative number gives us a p value
			//that approaches 2. 
			return 0.0;
			//System.err.println("An error occurred while calculating the PValue");
		}
		
		return pvalue;
	}
	
}
