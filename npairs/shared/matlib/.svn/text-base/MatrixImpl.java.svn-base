package npairs.shared.matlib;

import jmatlink.*;

/** This class implements a concrete instance of the abstract type Matrix.  
 *  Which concrete Matrix class is used depends on the underlying matrix library
 *  to be used; the matrix library is set via the MatrixImpl constructor.
 *  Later calls to MatrixImpl can be made via simplified (2 argument) constructor,
 *  as long as same matrix library is being used.
 *  
 *  Currently, MatrixImpl implements ColtMatrix and MatlabMatrix.  
 *  
 *  To add a new concrete type extension of Matrix as an option, just set value 
 *  of Matrix 'mat' to a new instance of the new concrete Matrix type within an
 *  'if (matlibType == "myNewMatLibType")' statement.
 *  
 *  Example usage implementing ColtMatrix:
 *  
 *  	MatrixImpl mImpl = new MatrixImpl(10, 10, "Colt");
 *  	Matrix myMat = mImpl.getMatrix();
 *  
 *  Now myMat is actually a ColtMatrix, and can call all Matrix methods 
 *  (as implemented in ColtMatrix).
 *    
 * @author anita
 *
 */       
public class MatrixImpl {

	private Matrix mat;
	//protected static JMatLink matlabEngine;
	//private static int currNumMatlabMatImplInstances = 0;
	private static String matlibType; // TODO: should be synchronized or protected in some other way?
	
	
	/** Constructs new instance of MatrixImpl.
	 * 
	 * @param nrows
	 * 			Number of rows in Matrix
	 * @param ncols
	 * 			Number of columns in Matrix
	 * @param matlibType
	 * 			Type of matrix library to be used (case insensitive).  
	 * 			Current implementation includes 'Matlab' and 'Colt'.
	 * 
	 * @see matlib.MatlabMatrix
	 * @see matlib.ColtMatrix
	 * @see matlib.Matrix
	 * @see #MatrixImpl(int, int)
	 * @see #MatrixImpl(double[][], String)
	 * 
	 * @throws MatrixException if matlibType is invalid
	 */
	public MatrixImpl (int nrows, int ncols, String matlibType) throws MatrixException {
		MatrixImpl.matlibType = matlibType;
		if (matlibType.toUpperCase().equals("COLT")) {
			mat = new ColtMatrix(nrows, ncols);
		}
		else if (matlibType.toUpperCase().equals("PARALLELCOLT")) {
			mat = new ParallelColtDoubleMatrix(nrows, ncols);
		}
		else if (matlibType.toUpperCase().equals("PARALLELCOLT (FLOAT)")) {
			mat = new ParallelColtFloatMatrix(nrows, ncols);
		}
		else if (matlibType.toUpperCase().equals("MATLAB")) {
			// open a new matlab engine if one hasn't been opened yet
//			if (currNumMatlabMatImplInstances == 0) {
//				System.out.println("Opening new matlab engine...");
//				matlabEngine = new JMatLink();
//				matlabEngine.engOpen();
//			}
//			currNumMatlabMatImplInstances += 1;
//			System.out.println("Adding matlab instance # " + 
//					currNumMatlabMatImplInstances + "...");
			mat = new MatlabMatrix(nrows, ncols);
		}
		else throw new MatrixException(matlibType + " - No such matrix library found.");
	}

	/** Constructs new instance of MatrixImpl.
	 * 
	 * @param contents
	 * 			2D array of doubles containing values to be contained
	 * 			in Matrix
	 * @param matlibType
	 * 			Type of matrix library to be used (case insensitive).  
	 * 			Current implementation includes 'Matlab' and 'Colt'.
	 * 
	 * @see matlib.MatlabMatrix
	 * @see matlib.ColtMatrix
	 * @see matlib.Matrix
	 * @see #MatrixImpl(int, int)
	 * @see #MatrixImpl(int, int, String)
	 * 
	 * @throws MatrixException if matlibType is invalid
	 */
	public MatrixImpl (double[][] contents, String matlibType) throws MatrixException {
		MatrixImpl.matlibType = matlibType;
		if (matlibType.toUpperCase().equals("COLT")) {
			mat = new ColtMatrix(contents);
		}
		else if (matlibType.toUpperCase().equals("PARALLELCOLT")) {
			mat = new ParallelColtDoubleMatrix(contents);
		}
		else if (matlibType.toUpperCase().equals("PARALLELCOLT (FLOAT)")) {
			mat = new ParallelColtFloatMatrix(contents);
		}
		else if (matlibType.toUpperCase().equals("MATLAB")) {
			// open a new matlab engine if one hasn't been opened yet
//			if (currNumMatlabMatImplInstances == 0) {
//				System.out.println("Opening new matlab engine...");
//				matlabEngine = new JMatLink();
//				matlabEngine.engOpen();
//			}
//			currNumMatlabMatImplInstances += 1;
//			System.out.println("Adding matlab instance # " + 
//					currNumMatlabMatImplInstances + "...");
			mat = new MatlabMatrix(contents);
		}
		else throw new MatrixException(matlibType + " - No such matrix library found.");
	}
	
	/** Constructs new instance of MatrixImpl.  
	 *  REQUIRED: matlibType has been initialized already. 
	 * 
	 * @param contents
	 * 			2D array of doubles containing values to be contained
	 * 			in Matrix
	 * @see matlib.MatlabMatrix
	 * @see matlib.ColtMatrix
	 * @see matlib.Matrix
	 * @see #MatrixImpl(int, int, String)
	 * @see #MatrixImpl(double[][], String)
	 * @see #MatrixImpl(int, int)
	 * 
	 */
	public MatrixImpl(double[][] contents) {
		if (matlibType.toUpperCase().equals("COLT")) {
			mat = new ColtMatrix(contents);
		}
		else if (matlibType.toUpperCase().equals("PARALLELCOLT")) {
			mat = new ParallelColtDoubleMatrix(contents);
		}
		else if (matlibType.toUpperCase().equals("PARALLELCOLT (FLOAT)")) {
			mat = new ParallelColtFloatMatrix(contents);
		}
		else if (matlibType.toUpperCase().equals("MATLAB")) {
			// open a new matlab engine if one hasn't been opened yet
//			if (currNumMatlabMatImplInstances == 0) {
//				System.out.println("Opening new matlab engine...");
//				matlabEngine = new JMatLink();
//				matlabEngine.engOpen();
//			}
//			currNumMatlabMatImplInstances += 1;
//			System.out.println("Adding matlab instance # " + 
//					currNumMatlabMatImplInstances + "...");
			mat = new MatlabMatrix(contents);
		}
		// else throws NullPointerException if matlibType not initialized already
	}
	
	
	/** Constructs new instance of MatrixImpl.  
	 *  REQUIRED: matlibType has been initialized already. 
	 * 
	 * @param nrows
	 * 			Number of rows in Matrix
	 * @param ncols
	 * 			Number of columns in Matrix
	 * 	
	 * @see matlib.MatlabMatrix
	 * @see matlib.ColtMatrix
	 * @see matlib.Matrix
	 * @see #MatrixImpl(int, int, String)
	 * @see #MatrixImpl(double[][], String)
	 * 
	 */
	public MatrixImpl (int nrows, int ncols) {
	
		if (matlibType.toUpperCase().equals("COLT")) {
			mat = new ColtMatrix(nrows, ncols);
		}
		else if (matlibType.toUpperCase().equals("PARALLELCOLT")) {
			mat = new ParallelColtDoubleMatrix(nrows, ncols);
		}
		else if (matlibType.toUpperCase().equals("PARALLELCOLT (FLOAT)")) {
			mat = new ParallelColtFloatMatrix(nrows, ncols);
		}
		else if (matlibType.toUpperCase().equals("MATLAB")) {
			// open a new matlab engine if one hasn't been opened yet
//			if (currNumMatlabMatImplInstances == 0) {
//				System.out.println("Opening new matlab engine...");
//				matlabEngine = new JMatLink();
//				matlabEngine.engOpen();
//			}
//			currNumMatlabMatImplInstances += 1;
//			System.out.println("Adding matlab instance # " + 
//					currNumMatlabMatImplInstances + "...");
			mat = new MatlabMatrix(nrows, ncols);
		}
		// else throws NullPointerException if matlibType not initialized already
	}
	
	
	/** Returns instance of concrete Matrix type implemented in this MatrixImpl.
	 *  
	 * @return implemented Matrix instance
	 */
	public Matrix getMatrix() {
		return mat;
	}
	
	
//	public static void setMatlibType(String matlibType) {
//		MatrixImpl.matlibType = matlibType;
//	}
	
	
//	public static void incrementCurrNumMLInst() {
//		System.out.println("Incrementing currNumMatlabMatImplInstances...");
//		MatrixImpl.currNumMatlabMatImplInstances += 1;
//		System.out.println("currNumMatlabMatImplInstances: " 
//				+ MatrixImpl.currNumMatlabMatImplInstances);
//	}
//	
//	public static void decrementCurrNumMLInst() {
//		System.out.println("Decrementing currNumMatlabMatImplInstances...");
//		MatrixImpl.currNumMatlabMatImplInstances -= 1;
//		System.out.println("currNumMatlabMatImplInstances: " 
//				+ MatrixImpl.currNumMatlabMatImplInstances);
//	}
	
//	public static void setMatlabEng(JMatLink eng) {
//		MatrixImpl.matlabEngine = eng;
//	}
	
	
//	protected void finalize() throws Throwable {
//		if (matlibType.toUpperCase().equals("MATLAB")) {
////			System.out.println("currNumMatlabMatImplInstances: " +
////					MatrixImpl.currNumMatlabMatImplInstances);
////			System.out.println("Decrementing currNumMatlabMatImplInstances");
////			currNumMatlabMatImplInstances -= 1;
//			if (currNumMatlabMatImplInstances == 0) {
//				System.out.println("Closing matlab engine...");
//				matlabEngine.engClose();
//			}
//		}
//	}
}