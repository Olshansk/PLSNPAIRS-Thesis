package pls.test;

import npairs.io.NpairsjIO;
import npairs.shared.matlib.ColtMatrix;
import npairs.shared.matlib.EigenvalueDecomposition;

public class EVDTest {
	
	public EVDTest() {
		
		// create matrix with zero rows
		double[][] matVals = {{0,0,0,0,0,0,0,0,0,0},
				{1,2,3,4,1,3,5,9,10,4},
				{1,2.2,2.9,4.01,1,3,5.3,9.1,10.4,4},
				{7,7,6,6,3,3,5,5,2,1},
				{0,0,0,0,0,0,0,0,0,0},
				{8,9,9,10,4,6,10,14,12,5}};							
		ColtMatrix M = new ColtMatrix(matVals);
		System.out.println("M: ");
		M.print();
		
		// create symmetric matrix MMt
		ColtMatrix MMt = M.sspByRow();
		System.out.println("MMt:");
		MMt.print();
		
		// do evd on MMt
		EigenvalueDecomposition evd = MMt.eigenvalueDecomposition();
		System.out.println("EVD evects: ");
		evd.getEvects().print();
		System.out.println("EVD evals: ");
		NpairsjIO.print(evd.getRealEvals());
		
		// calculate data in EVD space
		ColtMatrix S = new ColtMatrix(M.numRows(), M.numRows());
		for (int i = 0; i < M.numRows(); ++i) {
			S.setQuick(i, i, Math.sqrt(evd.getRealEvals()[i]));
		}
		ColtMatrix evdSpData = (ColtMatrix)evd.getEvects().mult(S);
		System.out.println("Input data in EVD space: ");
		evdSpData.print();		
	}
	
	public static void main (String[] args) {
		EVDTest test = new EVDTest();
	}

}
