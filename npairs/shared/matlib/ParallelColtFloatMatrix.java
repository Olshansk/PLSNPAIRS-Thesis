package npairs.shared.matlib;

import java.util.Random;

import cern.colt.matrix.tfloat.FloatFactory2D;
import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.colt.matrix.tfloat.algo.DenseFloatAlgebra;
import cern.colt.matrix.tfloat.impl.SparseFloatMatrix2D;
import cern.colt.matrix.tfloat.algo.decomposition.DenseFloatSingularValueDecompositionDC;
import cern.colt.matrix.tfloat.algo.decomposition.DenseFloatEigenvalueDecomposition;

public class ParallelColtFloatMatrix extends Matrix {

	private int numRows;
	private int numCols;
	protected FloatMatrix2D matrix;

	public ParallelColtFloatMatrix(int nRows, int nCols) {
		matrix = FloatFactory2D.dense.make(nRows, nCols);
		numRows = nRows;
		numCols = nCols;
	}
	public ParallelColtFloatMatrix(double[][] contents) {
		matrix = FloatFactory2D.dense.make(doubleToFloat(contents));
		numRows = matrix.rows();
		numCols = matrix.columns();
	}
	public ParallelColtFloatMatrix(FloatMatrix2D M) {
		matrix = M;
		numRows = matrix.rows();
		numCols = matrix.columns();
	}
	private float[][] doubleToFloat(double[][] input) {
		float[][] ret = new float[input.length][input[0].length];
		for(int j = 0; j < input.length; j++){
			for(int i = 0; i < input[0].length; i++){
				ret[j][i] = (float)input[j][i];
			}
		}
		return ret;
	}
	private double[][] floatToDouble(float[][] input) {
		double[][] ret = new double[input.length][input[0].length];
		for(int j = 0; j < input.length; j++){
			for(int i = 0; i < input[0].length; i++){
				ret[j][i] = input[j][i];
			}
		}
		return ret;
	}
	private double[] floatToDouble(float[] input) {
		double[] ret = new double[input.length];
		for(int i = 0; i < input.length; i++){
			ret[i] = input[i];
		}
		return ret;
	}
	public double get(int row, int col) {
		return matrix.get(row, col);
	}
	public double getQuick(int row, int col) {
		return matrix.getQuick(row, col);
	}
	public double[] getColumn(int col) {
		return floatToDouble(matrix.viewColumn(col).toArray());
	}
	public double[] getColumnQuick(int col) {
		double[] colData = new double[numRows];
		for (int i = 0; i < numRows; ++i) {
			colData[i] = matrix.getQuick(i, col);
		}
		return colData;
	}
	public double[] getRow(int row) {
		return floatToDouble(matrix.viewRow(row).toArray());
	}
	public double[] getRowQuick(int row) {
		double[] rowData = new double[numCols];
		for(int i = 0; i < numCols; ++i) {
			rowData[i] = matrix.getQuick(row, i);
		}
		return rowData;
	}
	public ParallelColtFloatMatrix plusEquals(Matrix B) {
		if((numRows != B.numRows()) || (numCols != B.numCols())) {
			throw new IllegalArgumentException("Cannot add Matrices of unequal size");
		}
		for(int row = 0; row < numRows; ++row) {
			for(int col = 0; col < numCols; ++col) {
				matrix.setQuick(row, col, (float)(matrix.getQuick(row, col) + B.getQuick(row, col)));
			}
		}
		return this;
	}
	public ParallelColtFloatMatrix minusEquals(Matrix B) {
		if((numRows != B.numRows()) || (numCols != B.numCols())) {
			throw new IllegalArgumentException("Cannot calculate difference of " +
			"Matrices of unequal size");
		}
		for(int row = 0; row < numRows; ++row) {
			for(int col = 0; col < numCols; ++col) {
				matrix.setQuick(row, col, (float)(matrix.getQuick(row, col) - B.getQuick(row, col)));
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
				matrix.setQuick(row, col, (float)(matrix.getQuick(row, col) - colMeans[col]));
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
	public ParallelColtFloatMatrix meanCentreColumns() {
		ParallelColtFloatMatrix meanCentredMat = new ParallelColtFloatMatrix(numRows, numCols);
		for(int col = 0; col < numCols; col++) {
			double currColMean = matrix.viewColumn(col).zSum() / numRows;
			for(int row = 0; row < numRows; ++row) {
				meanCentredMat.setQuick(row, col, (float)(matrix.getQuick(row, col) - currColMean));
			}
		}
		return meanCentredMat;
	}

	/** Constructs and returns new ParallelColtFloatMatrix containing correlation of 
	 * this Matrix and input Matrix. 
	 * REQUIRED: M is a ParallelColtFloatMatrix 
     * @param Matrix - must have same number of rows as this Matrix.
	 * @return Matrix containing Pearson's correlation coefficients of 
	 *  each column of input Matrix with each column of this Matrix.
	 *  Dimensions of returned Matrix: 
	 *  (no. cols in this Matrix) rows X (no. cols in input Matrix) cols
	 *  i.e., each column of returned Matrix contains correlations
	 *  between corresponding column of input Matrix and columns of this Matrix.
	 */
	public ParallelColtFloatMatrix correlate(Matrix M) {
		int nCols = M.numCols(); 
		int nRows = this.numCols; 
		if (numRows != M.numRows()) {
			throw new IllegalArgumentException("Matrices must have same" +
				" number of rows.");
		}
		ParallelColtFloatMatrix corrMat = this.meanCentreColumns().transpose().mult(
				M.meanCentreColumns());	
		double[] stdDevThis = this.stdDevCols(); 
		double[] stdDevM = ((ParallelColtFloatMatrix)M).stdDevCols(); 
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
		ParallelColtFloatMatrix meanCentredMat = this.meanCentreColumns();
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
	public ParallelColtFloatMatrix sspByRow() {
		return new ParallelColtFloatMatrix(matrix.zMult(
				matrix, null, 1, 0, false, true));
	}
	
	
	/**  Constructs and returns new 
	 * 	DenseDoubleMatrix2D containing result of matrix multiplication M' * M
	 *  (where M is this ParallelColtMatrix, and M' == transpose of M).  
	 *  (SSP stands for 'sums of squares and products')
	 *  M is NOT mean-centred before multiplication.
	 *  
	 * @return M' * M, where M is this matrix, * is matrix multiplication 
	 *         and M' is the transpose of M
	 */
	public ParallelColtFloatMatrix sspByCol() {
		return new ParallelColtFloatMatrix(matrix.zMult(
				matrix, null, 1, 0, true, false));
	}
	/** Required: B is a ParallelColtMatrix.
	 * 
	 */
	public ParallelColtFloatMatrix mult(Matrix B) {
		// note could use cern.colt.matrix.linalg.Algebral.mult(A,B) instead
		return new ParallelColtFloatMatrix(matrix.zMult(
				((ParallelColtFloatMatrix)B).matrix, null, 1, 0, false, false));
	}
	
	public ParallelColtFloatMatrix mult(double scalar) {
		ParallelColtFloatMatrix result = new ParallelColtFloatMatrix(numRows, numCols);
		for(int row = 0; row < numRows; ++row) {
			for(int col = 0; col < numCols; ++col) {
				result.setQuick(row, col, (matrix.getQuick(row, col) * scalar));
			}
		}
		return result;
	}
	
	public ParallelColtFloatMatrix multInPlace(double scalar) {
		for (int row = 0; row < numRows; ++row) {
			for (int col = 0; col < numCols; ++col) {
				this.matrix.setQuick(row, col, 
					(float)(matrix.getQuick(row, col) * scalar));
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
		matrix.set(row, col, (float)value);
	}
	
	/**TODO: consider how to deal with casting from double
	 *       to float here - do we have to worry about loss
	 *       of precision, or is the fact that this is a float
	 *       back-end to a generic Matrix class that accepts
	 *       doubles already relying on assumption that this 
	 *       won't be a problem?
	 */
	public void set(double value) {
		matrix.assign((float)value);
	}
	
	public void setQuick(int row, int col, double value) {
		matrix.setQuick(row, col, (float)value);
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
			matrix.setQuick(row, col, (float)values[row]);
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
			matrix.setQuick(row, col, (float)values[col]);
		}
	}
	public void setMatrix(double[][] values) {
		matrix.assign(doubleToFloat(values));
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
		return floatToDouble(matrix.toArray());
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
	public ParallelColtFloatMatrix transpose() {
		// note could use cern.colt.matrix.linalg.Algebra.transpose(A);
		// API describes them as equivalent
		return new ParallelColtFloatMatrix(matrix.viewDice().copy());
	}
	public void setRandom() {
		Random r = new Random();
		for(int row = 0; row < numRows; ++row) {
			for(int col = 0; col < numCols; ++col) {
				matrix.setQuick(row, col, r.nextFloat());
			}
		}
	}

	/** REQUIRED: Matrix has numRows >= numCols.
	 * 	Output evects and singular vals are given in DESCENDING order, i.e. dim 
	 *  corresponding to largest singular val is first.
	 */
	public npairs.shared.matlib.SingularValueDecomposition svd() {
		DenseFloatSingularValueDecompositionDC svd = new DenseFloatSingularValueDecompositionDC(matrix, true, true);

		ParallelColtFloatMatrix U = new ParallelColtFloatMatrix(svd.getU());
		ParallelColtFloatMatrix V = new ParallelColtFloatMatrix(svd.getV());
		ParallelColtFloatMatrix S = new ParallelColtFloatMatrix(svd.getS());
		double[] sVals = floatToDouble(svd.getSingularValues());

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
		DenseFloatEigenvalueDecomposition evd = new DenseFloatEigenvalueDecomposition(matrix);

		// (Real) Eigenvals are sorted into descending order.
		double[] realEvals = floatToDouble(evd.getRealEigenvalues().toArray());
		double[] imagEvals = floatToDouble(evd.getImagEigenvalues().toArray());
		int[] sortedIndexOrder = sortDescending(realEvals);
		permuteArray(imagEvals, sortedIndexOrder);

		ParallelColtFloatMatrix eigenVectors = new ParallelColtFloatMatrix(evd.getV());
		eigenVectors = (ParallelColtFloatMatrix)eigenVectors.permuteColumns(sortedIndexOrder);
		ParallelColtFloatMatrix realEvalMat = new ParallelColtFloatMatrix(realEvals.length, realEvals.length);
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
	private static ParallelColtFloatMatrix getInvSqrtMat(double[] diag_values) {
		FloatMatrix2D invSqrtMat = new SparseFloatMatrix2D(diag_values.length, diag_values.length);
		double currAbsVal;
		for(int i = 0; i < diag_values.length; ++i) {
			currAbsVal = Math.abs(diag_values[i]);
			if(currAbsVal > 0) {
				invSqrtMat.setQuick(i, i, (float)(1. / Math.sqrt(currAbsVal))); 	
			}
			else {
				invSqrtMat.setQuick(i, i, 0);					
			}				
		}
		return new ParallelColtFloatMatrix(invSqrtMat);
	}


	/**
	 *  Returns contents of this ParallelColtMatrix as Colt FloatMatrix2D object.
	 * 
	 * @return Contents of this ParallelColtMatrix as Colt FloatMatrix2D object
	 */
	public FloatMatrix2D getColtMat() {
		return matrix;
	}

	/**
	 * Returns deep copy of this ParallelColtMatrix
	 * 
	 */
	public ParallelColtFloatMatrix copy() {
		ParallelColtFloatMatrix matCopy = new ParallelColtFloatMatrix(matrix.copy());
		return matCopy;
	}
	

	public ParallelColtFloatMatrix subMatrix(int[] rowRange, int[] colRange) {
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
		ParallelColtFloatMatrix subMat = new ParallelColtFloatMatrix(nRows, nCols);
		for (int row = 0; row < nRows; ++row) {
			for (int col = 0; col < nCols; ++col) {
				subMat.matrix.setQuick(row, col, matrix.getQuick(rowRange[0] + row, colRange[0] + col));
			}
		}

		return subMat;
	}

	public ParallelColtFloatMatrix subMatrixRows(int[] rowIndices) {
		ParallelColtFloatMatrix subMat = new ParallelColtFloatMatrix(rowIndices.length, numCols);
		for (int row = 0; row < rowIndices.length; ++row) {
			subMat.setRow(row, this.getRow(rowIndices[row]));
		}
		return subMat;
	}

	public ParallelColtFloatMatrix subMatrixCols(int[] colIndices) {
		ParallelColtFloatMatrix subMat = new ParallelColtFloatMatrix(numRows, colIndices.length);
		for (int col = 0; col < colIndices.length; ++col) {
			subMat.setColumn(col, this.getColumn(colIndices[col]));
		}
		return subMat;
	}

    // added by Grigori on Feb 25, 2009
    public double det() {
            DenseFloatAlgebra alg = new DenseFloatAlgebra();
            return alg.det(matrix);
    }
    	
	public ParallelColtFloatMatrix inverse() {
		// note Algebra constructor can be called with no arg; then equality tolerance
		// is given by Property.DEFAULT; otherwise use double equality tolerance as arg.
		DenseFloatAlgebra alg = new DenseFloatAlgebra();
		ParallelColtFloatMatrix inverseMat = new ParallelColtFloatMatrix(alg.inverse(matrix));
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
	public ParallelColtFloatMatrix permuteColumns(int[] colIndexOrder) {
		ParallelColtFloatMatrix permutedMat = new ParallelColtFloatMatrix(numRows, numCols);
		for (int i = 0; i < numCols; ++i) {
			permutedMat.setColumn(i, this.getColumn(colIndexOrder[i]));
		}
		return permutedMat;
	}

	public ParallelColtFloatMatrix zeroPadRows(int nRows, int[] rowInds) {
		if (nRows < this.numRows) {
			throw new IllegalArgumentException("Input no. of rows is too small.");
		}
		if (rowInds.length != this.numRows) {
			throw new IllegalArgumentException("Input array of row indices is not the right length.");
		}

		ParallelColtFloatMatrix paddedMat = new ParallelColtFloatMatrix(nRows, this.numCols);
		for (int r = 0; r < this.numRows; ++r) {
			paddedMat.setRowQuick(rowInds[r], this.getRow(r));
		}

		return paddedMat;
	}
	  /** Returns new ParellelColtFloatMatrix contained squared elements of this Matrix.
	 *  Squaring is done element-by-element.
	 *  @return new Matrix containing this Matrix squared 
	 *          (element-by-element)
	 */
    public ParallelColtFloatMatrix squareElements() {
    	ParallelColtFloatMatrix squaredMat = new ParallelColtFloatMatrix(this.numRows, 
    			this.numCols);
    	for (int r = 0; r < this.numRows; ++r) {
    		for (int c = 0; c < this.numCols; ++c) {
    			squaredMat.setQuick(r, c, Math.pow(this.getQuick(r,c), 2));
    		}
    	}
    	return squaredMat;
    }
	
}