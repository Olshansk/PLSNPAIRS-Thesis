package pls.analysis;

import Jama.Matrix;
import pls.shared.MLFuncs;

public class RRIXCorr {
	
	protected Matrix result = null;
	
	public RRIXCorr(Matrix des, Matrix dat) {
		Matrix design = des.copy();
		Matrix datamat = dat.copy();
		int m = datamat.getRowDimension();
		int n = datamat.getColumnDimension();
		
		Matrix avg = MLFuncs.columnMean(datamat);
		Matrix stdev = MLFuncs.std(datamat);
		
		int[] checknan = MLFuncs.find(stdev.getArray()[0], 0);

		for(int c : checknan) {
			for(int i = 0; i < m; i++) {
				datamat.set(i, c, 0);
			}
			
			avg.set(0, c, 0);
			stdev.set(0, c, 1);
		}
		
		for(int i = 0; i < m; i++) {
			Matrix temp = datamat.getMatrix(i, i, 0, n - 1).minus(avg).arrayRightDivide(stdev);
			datamat.setMatrix(i, i, 0, n - 1, temp);
		}
		
		int dm = design.getRowDimension();
		int dn = design.getColumnDimension();
		
		Matrix davg = MLFuncs.columnMean(design);
		Matrix dstdev = MLFuncs.std(design);
		
		int[] dchecknan = MLFuncs.find(dstdev.getArray()[0], 0);
		
		for(int dc : dchecknan) {
			for(int i = 0; i < dm; i++) {
				design.set(i, dc, 0);
			}
			
			davg.set(0, dc, 0);
			dstdev.set(0, dc, 1);
		}
		
		for(int i = 0; i < dm; i++) {
			Matrix temp = design.getMatrix(i, i, 0, dn - 1).minus(davg).arrayRightDivide(dstdev);
			design.setMatrix(i, i, 0, dn - 1, temp);
		}
		
		Matrix xProd = design.transpose().times(datamat);
		result = xProd.times(1.0 / (m - 1));
	}
}
