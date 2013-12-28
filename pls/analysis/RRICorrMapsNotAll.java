package pls.analysis;

import Jama.Matrix;
import pls.shared.MLFuncs;

public class RRICorrMapsNotAll {
	
	protected Matrix maps = null;
	
	/**
	 * Computes brain-behavior correlations on a subset of scans.
	 * The scan numbers containing behavior measures should be stored as a vector in 'bscans'
	 * @param behav
	 * @param datamat
	 * @param n
	 * @param bscans
	 * @return
	 */
	public RRICorrMapsNotAll(Matrix behav, Matrix datamat, int n, int[] bscans) {
		for(int i : bscans) {
			int[] range = MLFuncs.range(n * i, n * (i + 1) - 1);
			Matrix temp = new RRIXCorr(MLFuncs.getRows(behav, range), MLFuncs.getRows(datamat, range)).result;
			maps = MLFuncs.append(maps, temp);
		}
	}
}
