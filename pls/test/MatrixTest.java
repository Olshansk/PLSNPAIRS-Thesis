package pls.test;

import npairs.shared.matlib.*;

public class MatrixTest {

	// test Matrix.setDiag()
	public MatrixTest() {
		String[] matlibTypes = {"Colt", "ParallelColt", "ParallelColt (Float)"};
		int sz = 5;
		for (int i = 0; i < matlibTypes.length; ++i) {
			System.out.println("Using matlib: " + matlibTypes[i]);
		try {
			Matrix testMat = new MatrixImpl(5, 5, matlibTypes[i]).getMatrix();
			double[] diagEls = {3.4, 6.8, 7.4, -0.3, 9};
			double[] diagElsTooBig = {3.4, 6.8, 7.4, -0.3, 9, 43.5};
			double[] diagElsTooSmall = {3.4, 6.8, 7.4, -0.3};
			
			System.out.println("testMat after initialization: ");
			testMat.print();
			testMat.setDiag(diagEls);
			System.out.println("testMat after setting diag with setDiag:");
			testMat.print();
			
			Matrix testMat2 = new MatrixImpl(5, 5, matlibTypes[i]).getMatrix();
			System.out.println("testMat2 after initialization: ");
			testMat2.print();
//			testMat2 = testMat2.diag(diagEls);
			System.out.println("testMat2 after setting diag with diag:");
			testMat2.print();
				
			testMat.setRandom();
			System.out.println("testMat after setting to random matrix: ");
			testMat.print();
			testMat.setDiag(diagEls);
			System.out.println("testMat after setting diag with setDiag:");
			testMat.print();
			
			testMat2.setRandom();
			System.out.println("testMat2 after setting to random matrix: ");
			testMat2.print();
//			testMat2 = testMat2.diag(diagEls);
			System.out.println("testMat2 after setting diag with diag:");
			testMat2.print();
				
			
			
//			testMat.setDiag(diagElsTooBig);
//			System.out.println("testMat after setting diag (too big):");
//			testMat.print();
//			testMat.setDiag(diagElsTooSmall);
//			System.out.println("testMat after setting diag (too small):");
//			testMat.print();
		}
		catch (MatrixException me) {
			me.printStackTrace();
		}
		}
	}
	
	public static void main(String[] args) {
		MatrixTest mTest = new MatrixTest();
	}
}
