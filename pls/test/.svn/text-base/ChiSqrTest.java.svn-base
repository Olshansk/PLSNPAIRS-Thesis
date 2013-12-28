package pls.test;

import org.apache.commons.math.MathException;
import org.apache.commons.math.distribution.ChiSquaredDistributionImpl;

public class ChiSqrTest {

	private ChiSquaredDistributionImpl chiSqr;
	
	public ChiSqrTest(double dof, double alpha) throws MathException {
		chiSqr = new ChiSquaredDistributionImpl(dof);
		double upperCritVal = chiSqr.inverseCumulativeProbability(alpha);
		System.out.println("Upper critical value for alpha = " + alpha + ": " 
				+  upperCritVal);
		double prob = chiSqr.cumulativeProbability(upperCritVal);
		System.out.println("Calculated cumul. prob. for crit val " + upperCritVal + ": "
				+ prob);
		
	}
	public static void main (String[] args) {
//		double dof = Double.parseDouble(args[0]);
//		double alpha = Double.parseDouble(args[1]);
		double dof = 2;
		double[] alpha = {0.95,0.99,0.999,0.9999,0.99999,0.999999,0.9999999,0.99999999,
				0.99999999999999};
		
		try {
			for (int i = 0; i < alpha.length; ++i) {
				ChiSqrTest chiSqrTest = new ChiSqrTest(dof, alpha[i]);
			}
		} catch (MathException me) {
			me.printStackTrace();
		}
	}
}
