package npairs.shared.matlib;

import java.util.Random;

import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.colt.matrix.linalg.EigenvalueDecomposition;
import cern.colt.matrix.linalg.SingularValueDecomposition;

public class ColtMatrix extends Matrix {

	private int numRows;
	private int numCols;
	protected DoubleMatrix2D coltMat;
	
	public ColtMatrix(int nRows, int nCols) {
            coltMat = new DenseDoubleMatrix2D(nRows, nCols);
        	numRows = nRows;
        	numCols = nCols;
	}
	
	public ColtMatrix(double[][] contents) {
			coltMat = new DenseDoubleMatrix2D(contents);
			numRows = coltMat.rows();
			numCols = coltMat.columns();
	}
	
	public ColtMatrix(DoubleMatrix2D M) {
		    coltMat =  M;
		    numRows = coltMat.rows();
		    numCols = coltMat.columns();
	}
	
	public double get(int row, int col) {
		return coltMat.get(row, col);
	}
	
	public double getQuick(int row, int col) {
		return coltMat.getQuick(row, col);
	}

	public double[] getColumn(int col) {
		return coltMat.viewColumn(col).toArray();
	}
	
	public double[] getColumnQuick(int col) {
		double[] colData = new double[numRows];
		for (int i = 0; i < numRows; ++i) {
			colData[i] = coltMat.getQuick(i, col);
		}
		return colData;
	}

	public double[] getRow(int row) {
		return coltMat.viewRow(row).toArray();
	}
	
	public double[] getRowQuick(int row) {
		double[] rowData = new double[numCols];
		for (int i = 0; i < numCols; ++i) {
			rowData[i] = coltMat.getQuick(row, i);
		}
		return rowData;
	}
	
	public ColtMatrix plusEquals(Matrix B) {
		if ((numRows != B.numRows()) || (numCols != B.numCols())) {
			throw new IllegalArgumentException("Cannot add Matrices of unequal size");
		}
		for (int row = 0; row < numRows; ++row) {
			for (int col = 0; col < numCols; ++col) {
				coltMat.setQuick(row, col, coltMat.getQuick(row, col) + B.getQuick(row, col));
			}
		}
		return this;
	}
	
	public ColtMatrix minusEquals(Matrix B) {
		if ((numRows != B.numRows()) || (numCols != B.numCols())) {
			throw new IllegalArgumentException("Cannot calculate difference of " +
					"Matrices of unequal size");
		}
		for (int row = 0; row < numRows; ++row) {
			for (int col = 0; col < numCols; ++col) {
				coltMat.setQuick(row, col, coltMat.getQuick(row, col) - B.getQuick(row, col));
			}
		}
		return this;
	}

	/** Mean-centres columns of this ColtMatrix in place.
	 *  This ColtMatrix is modified.
	 */
	public double[] meanCentreColumnsInPlace() {
		double[] colMeans = new double[numCols];
			for (int col = 0; col < numCols; col++) {
			    colMeans[col] = coltMat.viewColumn(col).zSum() / numRows;
			    for (int row = 0; row < numRows; ++row) {
			    	coltMat.setQuick(row, col, coltMat.getQuick(row, col) - colMeans[col]);
				}
			}
			return colMeans;
	}
	
    public double[] colMeans() {
    	double[] colMeans = new double[numCols];
		for (int col = 0; col < numCols; col++) {
		    colMeans[col] = coltMat.viewColumn(col).zSum() / numRows;
		}
		return colMeans;
    }
	/** Constructs and returns new ColtMatrix containing mean-centred
	 *  columns of this ColtMatrix.
	 *  This ColtMatrix is not modified.
	 */
	public ColtMatrix meanCentreColumns() {
		ColtMatrix meanCentredMat = new ColtMatrix(numRows, numCols);
		for (int col = 0; col < numCols; col++) {
			    double currColMean = coltMat.viewColumn(col).zSum() / numRows;
			    for (int row = 0; row < numRows; ++row) {
			    	meanCentredMat.coltMat.setQuick(row, col, coltMat.getQuick(row, col) - currColMean);
				}
		}
		return meanCentredMat;
	}
	
	/** Constructs and returns new ColtMatrix containing correlation of 
	 * this Matrix and input Matrix. 
	 * REQUIRED: M is a ColtMatrix 
     * @param Matrix - must have same number of rows as this Matrix.
	 * @return Matrix containing Pearson's correlation coefficients of 
	 *  each column of input Matrix with each column of this Matrix.
	 *  Dimensions of returned Matrix: 
	 *  (no. cols in this Matrix) rows X (no. cols in input Matrix) cols
	 *  i.e., each column of returned Matrix contains correlations
	 *  between corresponding column of input Matrix and columns of this Matrix.
	 */
	public ColtMatrix correlate(Matrix M) {
		int nCols = M.numCols(); 
		int nRows = this.numCols; 
		if (numRows != M.numRows()) {
			throw new IllegalArgumentException("Matrices must have same" +
				" number of rows.");
		}
		ColtMatrix corrMat = this.meanCentreColumns().transpose().mult(
				M.meanCentreColumns());	
		double[] stdDevThis = this.stdDevCols(); 
		double[] stdDevM = ((ColtMatrix)M).stdDevCols(); 
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
		ColtMatrix meanCentredMat = this.meanCentreColumns();
		int dof = numRows - 1;
		for (int d = 0; d < numCols ; ++d) {
			double[] currCol = meanCentredMat.getColumnQuick(d);
			stddev[d] = Math.sqrt(dotProd(currCol, currCol) / dof); 							
		}
		return stddev;
	}
	

	/** Constructs and returns new 
	 * 	ColtMatrix containing result of matrix multiplication M * M'
	 *  (where M is this ColtMatrix, and M' == transpose of M).  
	 *  (SSP stands for 'sums of squares and products')
	 *  M is NOT mean-centred before multiplication.
	 * @return M * M', where M is this matrix, * is matrix multiplication 
	 *         and M' is the transpose of M
	 */		

	public ColtMatrix sspByRow() {
		return new ColtMatrix(coltMat.zMult(coltMat, null, 1, 0, false, true));	
	}
		
	/** Constructs and returns new 
	 * 	DenseDoubleMatrix2D containing result of matrix multiplication M' * M
	 *  (where M is this ColtMatrix, and M' == transpose of M).  
	 *  (SSP stands for 'sums of squares and products')
	 *  M is NOT mean-centred before multiplication.
	 *  
	 * @return M' * M, where M is this matrix, * is matrix multiplication 
	 *         and M' is the transpose of M
	 */		

	public ColtMatrix sspByCol() {
		return new ColtMatrix(coltMat.zMult(coltMat, null, 1, 0, true, false));
	}
		
	/** Required: B is a ColtMatrix.
	 * 
	 */
	public ColtMatrix mult(Matrix B) {
		// note could use cern.colt.matrix.linalg.Algebral.mult(A,B) instead
		return new ColtMatrix(coltMat.zMult(((ColtMatrix)B).coltMat, null, 
							1, 0, false, false));
	}
	
	public ColtMatrix multInPlace(double scalar) {
		for (int row = 0; row < numRows; ++row) {
			for (int col = 0; col < numCols; ++col) {
				this.coltMat.setQuick(row, col, (coltMat.getQuick(row, col) * scalar));
			}
		}
		return this;
	}
	
	public ColtMatrix mult(double scalar) {
		ColtMatrix result = new ColtMatrix(numRows, numCols);
		for (int row = 0; row < numRows; ++row) {
			for (int col = 0; col < numCols; ++col) {
				result.coltMat.setQuick(row, col, (coltMat.getQuick(row, col) * scalar));
			}
		}
		return result;
	}
		
    public int numCols() {
		return numCols;
	}

	public int numRows() {
		return numRows;
	}

	public void set(int row, int col, double value) {
        coltMat.set(row, col, value);
	}
	
	public void set(double value) {
		coltMat.assign(value);
	}
	
	public void setQuick(int row, int col, double value) {
		coltMat.setQuick(row, col, value);
	}

	public void setColumn(int col, double[] values) {
		// check size
		if (col < 0 || col >= numCols) {
			throw new IllegalArgumentException("Input column index out of bounds");
		}
		if (values.length != numRows) {
			throw new IllegalArgumentException("Size of input array does not " +
					"match size of Matrix");
		}
		setColumnQuick(col, values);
	}
	
	public void setColumnQuick(int col, double[] values) {
		for (int row = 0; row < numRows; ++row) {
			coltMat.setQuick(row, col, values[row]);
		}
	}

	public void setRow(int row, double[] values) {
		// check size
		if (row < 0 || row >= numRows) {
			throw new IllegalArgumentException("Input row index out of bounds");
		}
		if (values.length != numCols) {
			throw new IllegalArgumentException("Size of input array does not " +
					"match size of Matrix");
		}
		setRowQuick(row, values);
	}
	
	public void setRowQuick(int row, double[] values) {
		for (int col = 0; col < numCols; ++col) {
			coltMat.setQuick(row, col, values[col]);
		}
	}
	
	public void setMatrix(double[][] values) {
			coltMat.assign(values);
	}
	
	public void setSubMatrix(Matrix subM, int firstRowIdx, int firstColIdx) {
		int lastRow = firstRowIdx + subM.numRows();
		int lastCol = firstColIdx + subM.numCols();
		if  ((this.numRows < lastRow) 
				||
			 (this.numCols < lastCol)) {
			throw new IllegalArgumentException("Cannot embed given Matrix into "
					+ "this Matrix at given location: incompatible sizes"
					+ "\n(Last row, col = " + lastRow +  ", " + lastCol + "; Matrix size = "
					+ "[" + this.numRows + " X " + this.numCols + "])");
			
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
		return coltMat.toArray();
	}
	
	public double[] toRowPacked1DArray() {
		double[] contents1D = new double[coltMat.size()];
		for (int i = 0; i < coltMat.rows(); ++i) {
			for (int j = 0; j < coltMat.columns(); ++j) {
				contents1D[i * coltMat.columns() + j] = coltMat.getQuick(i, j);
			}
		}
		return contents1D;
	}

	/** Constructs and returns new ColtMatrix containing transpose of this
	 *  ColtMatrix.  
	 *  
	 *  @return transpose of this ColtMatrix
	 */
	public ColtMatrix transpose() {
		// note could use cern.colt.matrix.linalg.Algebra.transpose(A);
		// API describes them as equivalent
		return new ColtMatrix(coltMat.viewDice());
	}	
	
	public void setRandom() {
		Random r = new Random();
		for (int row = 0; row < numRows; ++row) {
			for (int col = 0; col < numCols; ++col) {
				this.coltMat.setQuick(row, col, r.nextDouble());
			}
		}
	}
	
	
	/** REQUIRED: Matrix has numRows >= numCols.
	 * 	Output evects and singular vals are given in DESCENDING order, i.e. dim 
	 *  corresponding to largest singular val is first.
	 */
	public npairs.shared.matlib.SingularValueDecomposition svd() {
		SingularValueDecomposition svd = new SingularValueDecomposition(coltMat);
		
		ColtMatrix U = new ColtMatrix(svd.getU());
		ColtMatrix V = new ColtMatrix(svd.getV());
		ColtMatrix S = new ColtMatrix(svd.getS());
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
		EigenvalueDecomposition coltEVD = new EigenvalueDecomposition(coltMat);
		
		// (Real) Eigenvals are sorted into descending order.
		double[] realEvals = coltEVD.getRealEigenvalues().toArray();
		double[] imagEvals = coltEVD.getImagEigenvalues().toArray();
//		System.out.println("Real evals before sorting: ");
//		utils_tests.PCATest.printArray(realEvals);
		int[] sortedIndexOrder = sortDescending(realEvals);
//		System.out.println("After sorting: ");
//		utils_tests.PCATest.printArray(realEvals);
//		System.out.println("Sorting index array: ");
//		utils_tests.QuickTests.printArray(sortedIndexOrder);
		permuteArray(imagEvals, sortedIndexOrder);
		
		ColtMatrix eigenVectors = new ColtMatrix(coltEVD.getV());
//		System.out.println("Colt eigenvectors: ");
//		eigenVectors.print();
		eigenVectors = (ColtMatrix)eigenVectors.permuteColumns(sortedIndexOrder);
//		System.out.println("Colt eigenvectors after permutation: ");
//		eigenVectors.print();
		ColtMatrix realEvalMat = new ColtMatrix(realEvals.length, realEvals.length);
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
	
    /** Constructs and returns new ColtMatrix containing inverse square roots of absolute
     *  values of input diag_values down the main diagonal, and zeros elsewhere. 
	 * 
	 * @param diag_values - inverse square roots of the absolute value of these values will 
	 * 					    populate diagonal of returned matrix
	 * @return ColtMatrix containing SparseDoubleMatrix2D: diagonal matrix with
	 *         the inverse square roots of (absolute values of) input array values as 
	 *         diagonal elements.
	 *         Size of returned [square] matrix == length of input array X length of input array. 
	 */
	private static ColtMatrix getInvSqrtMat(double[] diag_values) {
		DoubleMatrix2D invSqrtMat = new SparseDoubleMatrix2D(diag_values.length, diag_values.length);
		double currAbsVal;
		for (int i = 0; i < diag_values.length; ++i) {
			currAbsVal = Math.abs(diag_values[i]);
			if (currAbsVal > 0) {
				invSqrtMat.setQuick(i, i, (1. / Math.sqrt(currAbsVal))); 	
		    }
			else {
				invSqrtMat.setQuick(i, i, 0);					
			}				
		}
		return new ColtMatrix(invSqrtMat);
	}

	
	/**
	 *  Returns contents of this ColtMatrix as Colt DoubleMatrix2D object.
	 * 
	 * @return
	 * 			Contents of this ColtMatrix as Colt DoubleMatrix2D object
	 */
	public DoubleMatrix2D getColtMat() {
		return coltMat;
	}
	
	/**
	 * Returns deep copy of this ColtMatrix
	 * 
	 */
	public ColtMatrix copy() {
		ColtMatrix matCopy = new ColtMatrix(this.coltMat.copy());
		return matCopy;
	}
	
	
    public ColtMatrix subMatrix(int[] rowRange, int[] colRange) {
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
    	ColtMatrix subMat = new ColtMatrix(nRows, nCols);
    	for (int row = 0; row < nRows; ++row) {
    		for (int col = 0; col < nCols; ++col) {
    			subMat.coltMat.setQuick(row, col, this.coltMat.getQuick(rowRange[0] + row, colRange[0] + col));
    		}
    	}
    	
    	return subMat;
    }
    
    public ColtMatrix subMatrixRows(int[] rowIndices) {
    	ColtMatrix subMat = new ColtMatrix(rowIndices.length, numCols);
		for (int row = 0; row < rowIndices.length; ++row) {
			subMat.setRow(row, this.getRow(rowIndices[row]));
		}
		return subMat;
	}
    
    public ColtMatrix subMatrixCols(int[] colIndices) {
    	ColtMatrix subMat = new ColtMatrix(numRows, colIndices.length);
		for (int col = 0; col < colIndices.length; ++col) {
			subMat.setColumn(col, this.getColumn(colIndices[col]));
		}
		return subMat;
	}
    
    // added by Grigori on Feb 25, 2009
    public double det() {
            Algebra alg = new Algebra();
            return alg.det(coltMat);
    }

    
    
    public ColtMatrix inverse() {
    	// note Algebra constructor can be called with no arg; then equality tolerance
    	// is given by Property.DEFAULT; otherwise use double equality tolerance as arg.
    	Algebra alg = new Algebra();
    	ColtMatrix inverseMat = new ColtMatrix(alg.inverse(coltMat));
    	return inverseMat;
    }
    
    public void setIdentity() {
    	coltMat.assign(0);
    	int lengthDiag = Math.min(numRows, numCols);
    	for (int i = 0; i < lengthDiag; ++i) {
    		coltMat.setQuick(i, i, 1);    		
    	}
    }
       
    /** Returns new ColtMatrix containing permutation of columns of this 
     * ColtMatrix according to input index array
     * (e.g. if ColtMatrix has 3 columns and index array is [1,0,2], then swap
     * first 2 columns of matrix.
     * REQUIRED: input index array has length == no. of cols in matrix.
     * 
     * @param colIndexOrder
     * @return Matrix with columns permuted according to colIndexOrder
     */
    public ColtMatrix permuteColumns(int[] colIndexOrder) {
    	ColtMatrix permutedMat = new ColtMatrix(numRows, numCols);
    	for (int i = 0; i < numCols; ++i) {
    		permutedMat.setColumn(i, this.getColumn(colIndexOrder[i]));
    	}
    	return permutedMat;
    }
    
    public ColtMatrix zeroPadRows(int nRows, int[] rowInds) {
    	if (nRows < this.numRows) {
			throw new IllegalArgumentException("Input no. of rows is too small.");
		}
		if (rowInds.length != this.numRows) {
			throw new IllegalArgumentException("Input array of row indices is not the right length.");
		}
    
		ColtMatrix paddedMat = new ColtMatrix(nRows, this.numCols);
    	for (int r = 0; r < this.numRows; ++r) {
    		paddedMat.setRowQuick(rowInds[r], this.getRow(r));
    	}
    	
    	return paddedMat;
    }
    
    /** Returns new ColtMatrix contained squared elements of this Matrix.
	 *  Squaring is done element-by-element.
	 *  @return this ColtMatrix squared (element-by-element)
	 */
    public ColtMatrix squareElements() {
    	ColtMatrix squaredMat = new ColtMatrix(this.numRows, this.numCols);
    	for (int r = 0; r < this.numRows; ++r) {
    		for (int c = 0; c < this.numCols; ++c) {
    			squaredMat.setQuick(r, c, Math.pow(this.getQuick(r,c), 2));
    		}
    	}
    	return squaredMat;
    }
}

