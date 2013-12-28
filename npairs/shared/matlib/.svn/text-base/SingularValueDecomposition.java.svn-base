package npairs.shared.matlib;

/**This class is essentially a structure for holding results from a
 * singular value decomposition.  All of its methods are 'get' methods for 
 * retrieving these results.  An instance of this class is returned by 
 * the svd() method in the abstract Matrix class.
 * All concrete implementations of Matrix must conduct the actual singular 
 * value decomposition and instantiate a SingularValueDecomposition object 
 * using the results within the svd() method. 
 * 
 *  svd[M] = U * S * transpose(V)
 * 
 * @author anita
 *
 */



public class SingularValueDecomposition {

	private Matrix U; // left eigenvectors from svd
	
	private Matrix V; // right eigenvectors from svd
	
	private Matrix S; // Matrix with singular values 
	                  // from svd in descending order 
	                  // along main diagonal 
	
	private double[] sVals; // singular values from svd
	                        // in descending order
	
	/** singular vals and evects should be in descending order.
	 * 
	 * @param U 
	 * 				Matrix containing left eigenvectors in
	 *              descending order
	 *              Given M = USVt, 
	 *              	dims U are 
	 *              		M.numRows() X M.numCols()
	 *            
	 * @param V
	 * 				Matrix containing right eigenvectors in
	 * 			    descending order
	 * 				Given M = USVt,
	 * 					dims V are
	 *  					M.numCols() X M.numCols()
	 * @param S
	 * 				Matrix containing singular values along
	 *              main diagonal, in descending order from
	 *              top to bottom
	 *              Given M = USVt,
	 *              	Dims S are
	 *              		M.numCols() X M.numCols()
	 *              
	 * @param sVals 
	 * 				double array containing singular values 
	 *              in descending order
	 * @see Matrix
	 */
	public SingularValueDecomposition(Matrix U, Matrix S, Matrix V, double[] sVals) {
		this.U = U;
		this.V = V;
		this.S = S;
		this.sVals = sVals;

	}
	
	public Matrix getU() {
		return U;
	}
	
	public Matrix getV() {
		return V;
	}
	
	public Matrix getS() {
		return S;
	}
	
	public double[] getSVals() {
		return sVals;
	}

	// note MatrixImpl.matlibType must be instantiated outside of this class
	public Matrix getInvSingValsMat() {
		Matrix invSVMat = new MatrixImpl(sVals.length, sVals.length).getMatrix();
		for (int i = 0; i < sVals.length; ++i) {
			invSVMat.setQuick(i, i, ( 1.0 / sVals[i]));
		}
		return invSVMat;
	}
	
}

