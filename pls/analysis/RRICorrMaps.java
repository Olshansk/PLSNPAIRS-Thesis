package pls.analysis;

import pls.shared.MLFuncs;
import Jama.Matrix;

public class RRICorrMaps {
	
	protected Matrix maps = null;
	
	public RRICorrMaps(Matrix behav, Matrix datamat, int n, int k) {
		for(int i = 0; i < k; i++) {
			int[] range = MLFuncs.range(n * i, n * (i + 1) - 1);
			Matrix temp = new RRIXCorr(MLFuncs.getRows(behav, range), MLFuncs.getRows(datamat, range)).result;
			maps = MLFuncs.append(maps, temp);
		}
	}
}
