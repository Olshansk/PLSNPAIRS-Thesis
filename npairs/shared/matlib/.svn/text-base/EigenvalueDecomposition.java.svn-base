package npairs.shared.matlib;

/**This class is essentially a structure for holding results from an 
 * eigenvalue decomposition.  All of its methods are 'get' methods for 
 * retrieving these results.  An instance of this class is returned by 
 * the eigenvalueDecomposition() method in the abstract Matrix class.
 * All concrete implementations of Matrix must conduct the actual eigenvalue
 * decomposition, and instantiate an EigenvalueDecomposition object using
 * the results, within the eigenvalueDecomposition() method.  
 * 
 * @author anita
 *
 */



public class EigenvalueDecomposition {

	private double[] realEvals;
//	 If evd has been done on symm. matrix, then all evals will be real.
//	 Otherwise, evals will be partitioned into real and imaginary parts.
	private double[] imagEvals;
	private Matrix evects;
	private Matrix invSqrtRealEvalMat;
	private Matrix realEvalMat;

	/** evals and evects should be in descending order.
	 * 
	 * @param realEvals 
	 * 				double array containing real parts of eigenvalues in
	 * 					descending order
	 * @param imagEvals 
	 * 				double array containing imaginary parts of eigenvalues 
	 * 					in order corresponding to descending order of real parts
	 * @param evects 
	 * 				Matrix containing eigenvectors in columns, in order
	 * 					corresponding to descending order of real parts of 
	 * 					eigenvalues
	 * @param invSqrtRealEvalMat
	 * 				diagonal Matrix containing inverse square roots of realEvals
	 * 					down main diagonal, in descending order
	 * @param realEvalMat
	 * 			    diagonal Matrix containing realEvals down main diagonal, in
	 * 					descending order
	 * @see Matrix
	 */
	public EigenvalueDecomposition(double[] realEvals, double[] imagEvals, Matrix evects,
			Matrix invSqrtRealEvalMat, Matrix realEvalMat) {
		this.realEvals = realEvals;
		this.imagEvals = imagEvals;
		this.evects = evects;
		this.invSqrtRealEvalMat = invSqrtRealEvalMat;
		this.realEvalMat = realEvalMat;
	}
	
	public double[] getRealEvals() {
		return realEvals;
	}
	
	/** Returns the first 'nDims' number of eigenvalues
	 * 
	 * @param nDims
	 * @return double array containing first 'nDims' evals
	 */
	public double[] getRealEvals(int nDims) {
		double[] subsetEvals = new double[nDims];
		for (int i = 0; i < nDims; ++i) {
			subsetEvals[i] = realEvals[i];
		}
		return subsetEvals;
	}
	
	public Matrix getRealEvalMat() {
		return realEvalMat;
	}
	
	/** Returns subMatrix containing first 'nDims' evals down
	 *  diagonal
	 * @param nDims
	 * @return subMatrix containing first 'nDims' evals down
	 *  diagonal
	 */
	public Matrix getRealEvalMat(int nDims) {
		int[] rowRange = {0, nDims - 1};
		int[] colRange = {0, nDims - 1};
		return realEvalMat.subMatrix(rowRange, colRange);
	}
	
	public double[] getImagEvals() {
		return imagEvals;
	}
	
	public Matrix getEvects() {
		return evects;
	}
	
	/** Returns subMatrix containing first 'nDims' eigenvectors,
	 *  i.e. first 'nDims' columns.
	 *  
	 * @param nDims
	 * @return Matrix containing only first 'nDims' eigenvectors (i.e., cols)
	 */
	public Matrix getEvects(int nDims) {
		int[] rowRange = {0, evects.numRows() - 1};
		int[] colRange = {0, nDims - 1};
		return evects.subMatrix(rowRange, colRange);
	}
	
	
	public Matrix getInvSqrtRealEvalMat() {
		return invSqrtRealEvalMat;
	}
	
	/**Returns square Matrix containing first 'nDims' inverse 
	 * square roots of real eigenvalues along diagonal
	 * 
	 * @param nDims
	 * @return Matrix containing first 'nDims' inverse
	 * square roots of real eigenvalues along diagonal
	 */
	public Matrix getInvSqrtRealEvalMat(int nDims) {
		int[] rowRange = {0, nDims - 1};
		int[] colRange = {0, nDims - 1};
		return invSqrtRealEvalMat.subMatrix(rowRange, colRange);
	}
	

}

