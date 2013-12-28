package pls.analysis;

import Jama.Matrix;
import pls.shared.MLFuncs;

public class RRITaskMean {
	
	protected Matrix taskMean = null;
	
	/**
	 * Returns a matrix of task means and standard deviations for an array
	 * with data for each task stacked on top of one another
	 * @param data
	 * @param n
	 * @return
	 */
	public RRITaskMean(Matrix data, int n) {
		int m1 = data.getRowDimension();
		int m = data.getColumnDimension();
		
		int k = m1 / n;
		
		Matrix meanMat = new Matrix(k, m);
		
		for(int i = 0; i < k; i++) {
			Matrix temp = data.getMatrix(n * i, n * (i + 1) - 1, 0, m - 1);
			if(temp.getRowDimension() == 1) {
				Matrix temp2 = new Matrix(1, m, MLFuncs.rowMean(temp).get(0, 0));
				meanMat.setMatrix(i, i, 0, m - 1, temp2);
			} else {
				meanMat.setMatrix(i, i, 0, m - 1, MLFuncs.columnMean(temp));
			}
		}
		taskMean = meanMat;
	}
	//added for PET (rri_task_mean1.m file)
	public RRITaskMean(Matrix data, int [] n) {
		int m1 = data.getRowDimension();
		int m = data.getColumnDimension();
		
		int k = n.length;
		
		Matrix meanMat = new Matrix(k, m);
		int accum =0;
		for(int i = 0; i < k; i++) {
			Matrix temp = data.getMatrix(accum+n[i], accum+1, 0, m - 1);
			if(temp.getRowDimension() == 1) {
				Matrix temp2 = new Matrix(1, m, MLFuncs.rowMean(temp).get(0, 0));
				meanMat.setMatrix(i, i, 0, m - 1, temp2);
			} else {
				meanMat.setMatrix(i, i, 0, m - 1, MLFuncs.columnMean(temp));
			}
			accum = accum + n[i];
		}
		taskMean = meanMat;
	}
}
