package pls.test;

import npairs.shared.matlib.Matrix;
import npairs.shared.matlib.MatrixException;
import npairs.shared.matlib.MatrixImpl;

public class CorrMatTest {

	public static void main(String[] args) {
		for (String a : args) {
			String matlibType = a;
			try {
				Matrix A = new MatrixImpl(4, 5, matlibType).getMatrix();
				Matrix B = new MatrixImpl(4,3).getMatrix();
				A.setRandom();
				B.setRandom();

				Matrix corrMat = A.correlate(B);
				System.out.println("A: ");
				A.print();
				System.out.println("B: ");
				B.print();
				System.out.println("Matlib type: " + matlibType);
				System.out.println("Correlation of A & B: ");
				corrMat.print();
			} 
			catch (MatrixException me) {
				me.printStackTrace();
			}
		}
	}
}
