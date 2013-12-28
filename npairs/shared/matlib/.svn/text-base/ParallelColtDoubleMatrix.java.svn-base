package npairs.shared.matlib;

import java.util.Random;

import cern.colt.matrix.tdouble.DoubleFactory2D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.decomposition.DenseDoubleSingularValueDecompositionDC;
import cern.colt.matrix.tdouble.algo.decomposition.DenseDoubleEigenvalueDecomposition;

public class ParallelColtDoubleMatrix extends Matrix {

	private int numRows;
	private int numCols;
	protected DoubleMatrix2D matrix;

	public ParallelColtDoubleMatrix(int nRows, int nCols) {
		matrix = DoubleFactory2D.dense.make(nRows, nCols);
		numRows = nRows;
		numCols = nCols;
	}
	
	public ParallelColtDoubleMatrix(double[][] contents) {
		matrix = DoubleFactory2D.dense.make(contents);
		numRows = matrix.rows();
		numCols = matrix.columns();
	}
	
	public ParallelColtDoubleMatrix(DoubleMatrix2D M) {
		matrix = M;
		numRows = matrix.rows();
		numCols = matrix.columns();
	}
	
	public double get(int row, int col) {
		return matrix.get(row, col);
	}
	
	public double getQuick(int row, int col) {
		return matrix.getQuick(row, col);
	}
	
	public double[] getColumn(int col) {
		return matrix.viewColumn(col).toArray();
	}
	
	public double[] getColumnQuick(int col) {
		double[] colData = new double[numRows];
		for (int i = 0; i < numRows; ++i) {
			colData[i] = matrix.getQuick(i, col);
		}
		return colData;
	}
	
	public double[] getRow(int row) {
		return matrix.viewRow(row).toArray();
	}
	
	public double[] getRowQuick(int row) {
		double[] rowData = new double[numCols];
		for(int i = 0; i < numCols; ++i) {
			rowData[i] = matrix.getQuick(row, i);
		}
		return rowData;
	}
	
	public ParallelColtDoubleMatrix plusEquals(Matrix B) {
		if((numRows != B.numRows()) || (numCols != B.numCols())) {
			throw new IllegalArgumentException("Cannot add Matrices of unequal size");
		}
		for(int row = 0; row < numRows; ++row) {
			for(int col = 0; col < numCols; ++col) {
				matrix.setQuick(row, col, (matrix.getQuick(row, col) + B.getQuick(row, col)));
			}
		}
		return this;
	}
	
	public ParallelColtDoubleMatrix minusEquals(Matrix B) {
		if((numRows != B.numRows()) || (numCols != B.numCols())) {
			throw new IllegalArgumentException("Cannot calculate difference of " +
			"Matrices of unequal size");
		}
		for(int row = 0; row < numRows; ++row) {
			for(int col = 0; col < numCols; ++col) {
				matrix.setQuick(row, col, (matrix.getQuick(row, col) - B.getQuick(row, col)));
			}
		}
		return this;
	}
	
	/** Mean-centres columns of this ParallelColtMatrix in place.
	 *  This ParallelColtMatrix is modified.
	 */
	public double[] meanCentreColumnsInPlace() {
		double[] colMeans = new double[numCols];
		for (int col = 0; col < numCols; col++) {
			colMeans[col] = matrix.viewColumn(col).zSum() / numRows;
			for (int row = 0; row < numRows; ++row) {
				matrix.setQuick(row, col, (matrix.getQuick(row, col) - colMeans[col]));
			}
		}
		return colMeans;
	}
	
	public double[] colMeans() {
		double[] colMeans = new double[numCols];
		for (int col = 0; col < numCols; col++) {
			colMeans[col] = matrix.viewColumn(col).zSum() / numRows;
		}
		return colMeans;
	}
	
	/** Constructs and returns new ParallelColtMatrix containing mean-centred
	 *  columns of this ParallelColtMatrix.
	 *  This ParallelColtMatrix is not modified.
	 */
	public ParallelColtDoubleMatrix meanCentreColumns() {
		ParallelColtDoubleMatrix meanCentredMat = new ParallelColtDoubleMatrix(numRows, numCols);
		for(int col = 0; col < numCols; col++) {
			double currColMean = matrix.viewColumn(col).zSum() / numRows;
			for(int row = 0; row < numRows; ++row) {
				meanCentredMat.setQuick(row, col, (float)(matrix.getQuick(row, col) - currColMean));
			}
		}
		return meanCentredMat;
	}
	
	
	/** Constructs and returns new ParallelColtDoubleMatrix containing correlation of 
	 * this Matrix and input Matrix. 
	 * REQUIRED: M is a ParallelColtDoubleMatrix 
     * @param Matrix - must have same number of rows as this Matrix.
	 * @return Matrix containing Pearson's correlation coefficients of 
	 *  each column of input Matrix with each column of this Matrix.
	 *  Dimensions of returned Matrix: 
	 *  (no. cols in this Matrix) rows X (no. cols in input Matrix) cols
	 *  i.e., each column of returned Matrix contains correlations
	 *  between corresponding column of input Matrix and columns of this Matrix.
	 */
	public ParallelColtDoubleMatrix correlate(Matrix M) {
		int nCols = M.numCols(); 
		int nRows = this.numCols; 
		if (numRows != M.numRows()) {
			throw new IllegalArgumentException("Matrices must have same" +
				" number of rows.");
		}
		ParallelColtDoubleMatrix corrMat = this.meanCentreColumns().transpose().mult(
				M.meanCentreColumns());	
		double[] stdDevThis = this.stdDevCols(); 
		double[] stdDevM = ((ParallelColtDoubleMatrix)M).stdDevCols(); 
		int dof = numRows - 1;
		for (int c = 0; c < nCols; ++c) {
			for (int r = 0; r < nRows; ++r) {
				double denom = stdDevThis[r] * stdDevM[c] * dof;
				double currCorr = corrMat.getQuick(r, c) / denom;
				corrMat.setQuick(r, c, currCorr);
			}
		}
		return corrMat;
	}
	
	/** Constructs and returns double array containing standard 
	 * deviation of columns of this Matrix
	 * @return
	 */
	private double[] stdDevCols() {
		double[] stddev = new double[numCols];
		ParallelColtDoubleMatrix meanCentredMat = this.meanCentreColumns();
		int dof = numRows - 1;
		for (int d = 0; d < numCols ; ++d) {
			double[] currCol = meanCentredMat.getColumnQuick(d);
			stddev[d] = Math.sqrt(dotProd(currCol, currCol) / dof); 							
		}
		return stddev;
	}

	/** Constructs and returns new 
	 * 	DenseDoubleMatrix2D containing result of matrix multiplication M * M'
	 *  (where M is this ColtMatrix, and M' == transpose of M).  
	 *  (SSP stands for 'sums of squares and products')
	 *  M is NOT mean-centred before multiplication.
	 *  
	 * @return M * M', where M is this matrix, * is matrix multiplication 
	 *         and M' is the transpose of M
	 */
	public ParallelColtDoubleMatrix sspByRow() {
		return new ParallelColtDoubleMatrix(matrix.zMult(
				matrix, null, 1, 0, false, true));
			
	}
	
	/** Constructs and returns new 
	 * 	DenseDoubleMatrix2D containing result of matrix multiplication M' * M
	 *  (where M is this ParallelColtMatrix, and M' == transpose of M).  
	 *  (SSP stands for 'sums of squares and products')
	 *  M is NOT mean-centred before multiplication.
	 *  
	 * @return M' * M, where M is this matrix, * is matrix multiplication 
	 *         and M' is the transpose of M
	 */
	public ParallelColtDoubleMatrix sspByCol() {
		return new ParallelColtDoubleMatrix(matrix.zMult(
				matrix, null, 1, 0, true, false));
	}
	
	/** Required: B is a ParallelColtMatrix.
	 * 
	 */
	public ParallelColtDoubleMatrix mult(Matrix B) {
		// note could use cern.colt.matrix.linalg.Algebral.mult(A,B) instead
		return new ParallelColtDoubleMatrix(matrix.zMult(
				((ParallelColtDoubleMatrix)B).matrix, null, 
				1, 0, false, false));
	}
	
	public ParallelColtDoubleMatrix mult(double scalar) {
		ParallelColtDoubleMatrix result = new ParallelColtDoubleMatrix(numRows, numCols);
		for(int row = 0; row < numRows; ++row) {
			for(int col = 0; col < numCols; ++col) {
				result.setQuick(row, col, (matrix.getQuick(row, col) * scalar));
			}
		}
		return result;
	}
	
	public ParallelColtDoubleMatrix multInPlace(double scalar) {
		for (int row = 0; row < numRows; ++row) {
			for (int col = 0; col < numCols; ++col) {
				this.matrix.setQuick(row, col, (matrix.getQuick(row, col) * scalar));
			}
		}
		return this;
	}
	
	public int numCols() {
		return numCols;
	}
	
	public int numRows() {
		return numRows;
	}
	
	public void set(int row, int col, double value) {
		matrix.set(row, col, value);
	}
	
	public void set(double value) {
		matrix.assign(value);
	}
	
	public void setQuick(int row, int col, double value) {
		matrix.setQuick(row, col, value);
	}
	
	public void setColumn(int col, double[] values) {
		// check size
		if(col < 0 || col >= numCols) {
			throw new IllegalArgumentException("Input column index out of bounds");
		}
		if(values.length != numRows) {
			throw new IllegalArgumentException("Size of input array does not " +
			"match size of Matrix");
		}
		setColumnQuick(col, values);
	}
	
	public void setColumnQuick(int col, double[] values) {
		for(int row = 0; row < numRows; ++row) {
			matrix.setQuick(row, col, values[row]);
		}
	}
	
	public void setRow(int row, double[] values) {
		// check size
		if(row < 0 || row >= numRows) {
			throw new IllegalArgumentException("Input row index out of bounds");
		}
		if(values.length != numCols) {
			throw new IllegalArgumentException("Size of input array does not " +
			"match size of Matrix");
		}
		setRowQuick(row, values);
	}
	
	public void setRowQuick(int row, double[] values) {
		for(int col = 0; col < numCols; ++col) {
			matrix.setQuick(row, col, values[col]);
		}
	}
	
	public void setMatrix(double[][] values) {
		matrix.assign(values);
	}
	
	public void setSubMatrix(Matrix subM, int firstRowIdx, int firstColIdx) {
		int lastRow = firstRowIdx + subM.numRows();
		int lastCol = firstColIdx + subM.numCols();
		if  ((this.numRows < lastRow) 
				||
			 (this.numCols < lastCol)) {
			throw new IllegalArgumentException("Cannot embed given Matrix into "
					+ "this Matrix at given location: incompatible sizes");
		}
		
		for (int r = firstRowIdx; r < lastRow; ++r) {
			for (int c = firstColIdx; c < lastCol; ++c) {
				this.setQuick(r, c, subM.getQuick(
						r - firstRowIdx, c - firstColIdx));
			}
		}
	}
	
	public void setSubMatrix(double[][] values, int firstRowIdx, int firstColIdx) {
		int lastRow = firstRowIdx + values.length;
		int lastCol = firstColIdx + values[0].length;
		if  ((this.numRows < values.length + firstRowIdx) 
				||
			 (this.numCols < values[0].length + firstColIdx)) {
			throw new IllegalArgumentException("Cannot embed given values into "
					+ "this Matrix at given location: incompatible sizes");
		}
		
		for (int r = firstRowIdx; r < lastRow; ++r) {
			for (int c = firstColIdx; c < lastCol; ++c) {
				this.setQuick(r, c, values[r - firstRowIdx][c - firstColIdx]);
			}
		}
	}
	
	public double[][] toArray() {
		return matrix.toArray();
	}
	
	public double[] toRowPacked1DArray() {
		double[] contents1D = new double[(int) matrix.size()];
		for(int i = 0; i < matrix.rows(); ++i) {
			for(int j = 0; j < matrix.columns(); ++j) {
				contents1D[i * matrix.columns() + j] = matrix.getQuick(i, j);
			}
		}
		return contents1D;
	}
	
	/** Constructs and returns new ParallelColtMatrix containing transpose of this
	 *  ParallelColtMatrix.
	 *  
	 *  @return transpose of this ParallelColtMatrix
	 */
	public ParallelColtDoubleMatrix transpose() {
		// note could use cern.colt.matrix.linalg.Algebra.transpose(A);
		// API describes them as equivalent
		return new ParallelColtDoubleMatrix(matrix.viewDice().copy());
	}
	
	public void setRandom() {
		Random r = new Random();
		for(int row = 0; row < numRows; ++row) {
			for(int col = 0; col < numCols; ++col) {
				matrix.setQuick(row, col, r.nextDouble());
			}
		}
	}

	/** REQUIRED: Matrix has numRows >= numCols.
	 * 	Output evects and singular vals are given in DESCENDING order, i.e. dim 
	 *  corresponding to largest singular val is first.
	 */
	public npairs.shared.matlib.SingularValueDecomposition svd() {
		DenseDoubleSingularValueDecompositionDC svd = new DenseDoubleSingularValueDecompositionDC(matrix, true, true);

		ParallelColtDoubleMatrix U = new ParallelColtDoubleMatrix(svd.getU());
		ParallelColtDoubleMatrix V = new ParallelColtDoubleMatrix(svd.getV());
		ParallelColtDoubleMatrix S = new ParallelColtDoubleMatrix(svd.getS());
		double[] sVals = svd.getSingularValues();

		npairs.shared.matlib.SingularValueDecomposition genericSVD =
			new npairs.shared.matlib.SingularValueDecomposition(
					U, S, V, sVals);
		return genericSVD;
	}
	
	/** REQUIRED: Matrix is square.
	 *  Output evects and evals are given in DESCENDING order, i.e. dim corresponding to
	 *  largest eval is first.
	 */
	public npairs.shared.matlib.EigenvalueDecomposition eigenvalueDecomposition() {
		DenseDoubleEigenvalueDecomposition evd = new DenseDoubleEigenvalueDecomposition(matrix);

		// (Real) Eigenvals are sorted into descending order.
		double[] realEvals = evd.getRealEigenvalues().toArray();
		double[] imagEvals = evd.getImagEigenvalues().toArray();
		int[] sortedIndexOrder = sortDescending(realEvals);
		permuteArray(imagEvals, sortedIndexOrder);

		ParallelColtDoubleMatrix eigenVectors = new ParallelColtDoubleMatrix(evd.getV());
		eigenVectors = (ParallelColtDoubleMatrix)eigenVectors.permuteColumns(sortedIndexOrder);
		ParallelColtDoubleMatrix realEvalMat = new ParallelColtDoubleMatrix(realEvals.length, realEvals.length);
		try {
			realEvalMat.setDiag(realEvals);
		}
		catch (MatrixException me) {
			// matrix is square by construction
		}

		npairs.shared.matlib.EigenvalueDecomposition genericEVD = 
			new npairs.shared.matlib.EigenvalueDecomposition(
					realEvals, imagEvals, eigenVectors, getInvSqrtMat(realEvals), realEvalMat);

		return genericEVD;
	}

	/** Constructs and returns new ParallelColtMatrix containing inverse square roots of absolute
	 *  values of input diag_values down the main diagonal, and zeros elsewhere. 
	 * 
	 * @param diag_values - inverse square roots of the absolute value of these values will 
	 * 					    populate diagonal of returned matrix
	 * @return ParallelColtMatrix containing SparseDoubleMatrix2D: diagonal matrix with
	 *         the inverse square roots of (absolute values of) input array values as 
	 *         diagonal elements.
	 *         Size of returned [square] matrix == length of input array X length of input array. 
	 */
	private static ParallelColtDoubleMatrix getInvSqrtMat(double[] diag_values) {
		DoubleMatrix2D invSqrtMat = new SparseDoubleMatrix2D(diag_values.length, diag_values.length);
		double currAbsVal;
		for(int i = 0; i < diag_values.length; ++i) {
			currAbsVal = Math.abs(diag_values[i]);
			if(currAbsVal > 0) {
				invSqrtMat.setQuick(i, i, (1. / Math.sqrt(currAbsVal))); 	
			}
			else {
				invSqrtMat.setQuick(i, i, 0);					
			}				
		}
		return new ParallelColtDoubleMatrix(invSqrtMat);
	}

	/**
	 *  Returns contents of this ParallelColtMatrix as Colt DoubleMatrix2D object.
	 * 
	 * @return Contents of this ParallelColtMatrix as Colt DoubleMatrix2D object
	 */
	public DoubleMatrix2D getColtMat() {
		return matrix;
	}

	/**
	 * Returns deep copy of this ParallelColtMatrix
	 * 
	 */
	public ParallelColtDoubleMatrix copy() {
		ParallelColtDoubleMatrix matCopy = new ParallelColtDoubleMatrix(matrix.copy());
		return matCopy;
	}
	

	public ParallelColtDoubleMatrix subMatrix(int[] rowRange, int[] colRange) {
		// TODO: can the following error code be incorporated into generic
		// Matrix class?
		if ((rowRange.length != 2) || (colRange.length != 2)) {
			throw new IllegalArgumentException("Input args must be 2-element arrays");
		}
		if ((rowRange[1] < rowRange[0]) || (colRange[1] < colRange[0])) {
			throw new IllegalArgumentException("Invalid row or col range");
		}
		if ((rowRange[0] < 0) || (rowRange[1] > this.numRows())
				|| (colRange[0] < 0) || (colRange[1] > this.numCols())) {
			throw new IllegalArgumentException("Row/col ranges must be subset of "
					+ "original Matrix");
		}

		int nRows = rowRange[1] - rowRange[0] + 1;
		int nCols = colRange[1] - colRange[0] + 1;
		ParallelColtDoubleMatrix subMat = new ParallelColtDoubleMatrix(nRows, nCols);
		for (int row = 0; row < nRows; ++row) {
			for (int col = 0; col < nCols; ++col) {
				subMat.matrix.setQuick(row, col, matrix.getQuick(rowRange[0] + row, colRange[0] + col));
			}
		}

		return subMat;
	}

	public ParallelColtDoubleMatrix subMatrixRows(int[] rowIndices) {
		ParallelColtDoubleMatrix subMat = new ParallelColtDoubleMatrix(rowIndices.length, numCols);
		for (int row = 0; row < rowIndices.length; ++row) {
			subMat.setRow(row, this.getRow(rowIndices[row]));
		}
		return subMat;
	}

	public ParallelColtDoubleMatrix subMatrixCols(int[] colIndices) {
		ParallelColtDoubleMatrix subMat = new ParallelColtDoubleMatrix(numRows, colIndices.length);
		for (int col = 0; col < colIndices.length; ++col) {
			subMat.setColumn(col, this.getColumn(colIndices[col]));
		}
		return subMat;
	}

    // added by Grigori on Feb 25, 2009
    public double det() {
            DenseDoubleAlgebra alg = new DenseDoubleAlgebra();
            return alg.det(matrix);
    }

	public ParallelColtDoubleMatrix inverse() {
		// note Algebra constructor can be called with no arg; then equality tolerance
		// is given by Property.DEFAULT; otherwise use double equality tolerance as arg.
		DenseDoubleAlgebra alg = new DenseDoubleAlgebra();
		ParallelColtDoubleMatrix inverseMat = new ParallelColtDoubleMatrix(alg.inverse(matrix));
		return inverseMat;
	}

	public void setIdentity() {
		matrix.assign(0);
		int lengthDiag = Math.min(numRows, numCols);
		for(int i = 0; i < lengthDiag; ++i) {
			matrix.setQuick(i, i, 1);    		
		}
	}

	/** Returns new ParallelColtMatrix containing permutation of columns of this 
	 * ParallelColtMatrix according to input index array
	 * (e.g. if ParallelColtMatrix has 3 columns and index array is [1,0,2], then swap
	 * first 2 columns of matrix.
	 * REQUIRED: input index array has length == no. of cols in matrix.
	 * 
	 * @param colIndexOrder
	 * @return Matrix with columns permuted according to colIndexOrder
	 */
	public ParallelColtDoubleMatrix permuteColumns(int[] colIndexOrder) {
		ParallelColtDoubleMatrix permutedMat = new ParallelColtDoubleMatrix(numRows, numCols);
		for (int i = 0; i < numCols; ++i) {
			permutedMat.setColumn(i, this.getColumn(colIndexOrder[i]));
		}
		return permutedMat;
	}

	public ParallelColtDoubleMatrix zeroPadRows(int nRows, int[] rowInds) {
		if (nRows < this.numRows) {
			throw new IllegalArgumentException("Input no. of rows is too small.");
		}
		if (rowInds.length != this.numRows) {			
			throw new IllegalArgumentException("Input array of row indices is not the right length.");
		}

		ParallelColtDoubleMatrix paddedMat = new ParallelColtDoubleMatrix(nRows, this.numCols);
		for (int r = 0; r < this.numRows; ++r) {
			paddedMat.setRowQuick(rowInds[r], this.getRow(r));
		}

		return paddedMat;
	}
	
	  /** Returns new ParellelColtDoubleMatrix contained squared elements of this Matrix.
	 *  Squaring is done element-by-element.
	 *  @return new Matrix containing this Matrix squared 
	 *          (element-by-element)
	 */
    public ParallelColtDoubleMatrix squareElements() {
    	ParallelColtDoubleMatrix squaredMat = new ParallelColtDoubleMatrix(this.numRows, 
    			this.numCols);
    	for (int r = 0; r < this.numRows; ++r) {
    		for (int c = 0; c < this.numCols; ++c) {
    			squaredMat.setQuick(r, c, Math.pow(this.getQuick(r,c), 2));
    		}
    	}
    	return squaredMat;
    }

}