package npairs.shared.matlib;

import jmatlink.*;
// Assumption: jmatlink engine has been opened outside of this class.  The opened engine is 
// included as an argument when constructing an instance of MatlabMatrix.

public class MatlabMatrix extends Matrix {

	private boolean debug = false;
	
	protected int numRows;
	protected int numCols;
	private static JMatLink engine = null;
	private static int matlabMatCount = 0; 	  // counts each instance of MatlabMatrix.
    
	protected String matlabMatContents;       // name of var containing matrix in 
	                                          // matlab workspace - must be unique for each 
	                                          // instance of MatlabMatrix.
	
	public MatlabMatrix (int nRows, int nCols) {
		matlabMatCount += 1;
		//System.out.println("Adding MatlabMatrix instance # " + matlabMatCount);
		matlabMatContents = "contents" + matlabMatCount; 
		numRows = nRows;
		numCols = nCols;
		if (engine == null) {
			System.out.println("Opening new matlab engine...");
			engine = new JMatLink();
			engine.engOpen();
		}
		engine.engPutArray(matlabMatContents, new double[nRows][nCols]);
	}
	
	public MatlabMatrix (double[][] contents) {
		matlabMatCount += 1;
//		if (debug) {
//			System.out.println("Adding MatlabMatrix instance # " + matlabMatCount);
//		}
		matlabMatContents = "contents" + matlabMatCount;
		numRows = contents.length;
		numCols = contents[0].length;
		if (engine == null) {
			if (debug) {
				System.out.println("Opening new matlab engine...");
			}
			engine = new JMatLink();
			engine.engOpen();
		}
		engine.engPutArray(matlabMatContents, contents);
	}
	
	public double getQuick(int row, int col) {
		String getCurrVal = matlabMatContents + "currVal = " + 
			matlabMatContents + "(" + (row + 1) + "," + (col + 1) + ");";
		engine.engEvalString(getCurrVal);
		
		double currVal = engine.engGetScalar(matlabMatContents + "currVal");
		return currVal;		
	}
	
	public double get(int row, int col) {
		if (row < 0 || row >= numRows) {
			throw new IndexOutOfBoundsException("Input row index out of bounds");
		}
		if (col < 0 || col >= numCols) {
			throw new IndexOutOfBoundsException("Input column index out of bounds");
		}
		return getQuick(row, col);
	}

	public double[] getColumn(int col) {
		if (col < 0 || col >= numCols) {
			throw new IndexOutOfBoundsException("Input column index out of bounds");
		}
		return getColumnQuick(col);
	}
	
	public double[] getColumnQuick(int col) {
		engine.engEvalString("currCol = " + matlabMatContents + "(:, " + (col + 1) + ");");
		double[][] currCol2D = engine.engGetArray("currCol");
		double[] currCol1D = trimTo1D(currCol2D);
		return currCol1D;
	}

	// XXX this should be static method, but where will it sit?  Need first to consider
	// the issue of how to store minimal-storage matrices, i.e. diagonal (sparse) and symmetric
	// matrices.  
//	public Matrix getInvSqrtMat(double[] diag_values, double zero_thresh) {
//		// TODO Auto-generated method stub
//		return null;
//	}

	public double[] getRow(int row) {
		if (row < 0 || row >= numRows) {
			throw new IndexOutOfBoundsException("Input row index out of bounds");
		}
		return getRowQuick(row);
    }
	
	public double[] getRowQuick(int row) {
		engine.engEvalString("currRow = " + matlabMatContents + "(" + (row + 1) + ", :);");
		double[][] currRow2D = engine.engGetArray("currRow");
		double[] currRow1D = trimTo1D(currRow2D);
		
		return currRow1D;
	}

	/** meanCentreColumnsInPlace(): Contents of this matrix are modified.
	 * 
	 * @return array of column means
	 */
	public double[] meanCentreColumnsInPlace() {
		String command = matlabMatContents + " = " + matlabMatContents + 
					     " - repmat(mean(" + matlabMatContents + "), " +
				         "size(" + matlabMatContents + ", 1), 1);";		
		engine.engEvalString(command);
		return this.colMeans();
	}

	public double[] colMeans() {
		String command = "colmeans = mean(" + matlabMatContents +")";
		engine.engEvalString(command);
		double[][] colMeans2D = engine.engGetArray("colmeans");
		return trimTo1D(colMeans2D);
	}

	/** Mean centres columns of this matrix and returns the result.  This
	 *  MatlabMatrix is NOT modified.
	 */
	public MatlabMatrix meanCentreColumns() {
		MatlabMatrix meanCentredMat = new MatlabMatrix(numRows, numCols);
		String command = meanCentredMat.matlabMatContents + " = " + 
			matlabMatContents + " - repmat(mean(" + matlabMatContents + "), " +
			"size(" + matlabMatContents + ", 1), 1);";
		engine.engEvalString(command);
		
		return meanCentredMat;
	}
	
	/** Calculates correlation of this Matrix and input Matrix.  
	 *  Required: Matrix M is a MatlabMatrix
	 * @param Matrix - must have same number of rows as this Matrix.
	 * @return Matrix containing Pearson's correlation coefficients of 
	 *  each column of input Matrix with each column of this Matrix.
	 *  Dimensions of returned Matrix: 
	 *  (no. cols in this Matrix) rows X (no. cols in input Matrix) cols
	 */
	public MatlabMatrix correlate(Matrix M) {
		MatlabMatrix corrMat = new MatlabMatrix(numCols, M.numCols());
			String command = corrMat.matlabMatContents + " = " +
				"corr(" + this.matlabMatContents + ", " + 
				((MatlabMatrix)M).matlabMatContents + ");";
			engine.engEvalString(command);
			
			return corrMat;
	}
	
	
	/** Required: Matrix B is a MatlabMatrix
	 * 
	 */
	public MatlabMatrix mult(Matrix B) {
		MatlabMatrix result = new MatlabMatrix(numRows, B.numCols());
		String command = result.matlabMatContents + " = " + matlabMatContents 
			+ " * " + ((MatlabMatrix)B).matlabMatContents + ";";
			engine.engEvalString(command);
	
		return result;
	}

	
	/** Multiplies every element in this Matrix by scalar and returns new Matrix 
	 *  containing result.
	 */
	
	public Matrix mult(double scalar) {
		MatlabMatrix result = new MatlabMatrix(numRows, numCols);
		engine.engPutArray("scalar", scalar);
		String command = result.matlabMatContents + " = " + matlabMatContents + 
				" *  scalar;";
		engine.engEvalString(command);
		
		// clean up matlab variable no longer needed
		engine.engEvalString("clear scalar");
		
		return result;		
	}
	
	public Matrix multInPlace(double scalar) {
		engine.engPutArray("scalar", scalar);
		String command = matlabMatContents + " = " + matlabMatContents + 
				" *  scalar;";
		engine.engEvalString(command);
		
		// clean up matlab variable no longer needed
		engine.engEvalString("clear scalar");
		
		return this;	
	}

	
	public MatlabMatrix plusEquals(Matrix B) {
		if ((numRows != B.numRows()) || (numCols != B.numCols())) {
			throw new IllegalArgumentException("Cannot add Matrices of unequal size");
		}
		String command = matlabMatContents + " = " + 
		matlabMatContents + " + " + ((MatlabMatrix)B).matlabMatContents + ";";
		engine.engEvalString(command);
		
		return this;
	}
	
	public MatlabMatrix minusEquals(Matrix B) {
		if ((numRows != B.numRows()) || (numCols != B.numCols())) {
			throw new IllegalArgumentException("Cannot add Matrices of unequal size");
		}
		String command = matlabMatContents + " = " + 
		matlabMatContents + " - " + ((MatlabMatrix)B).matlabMatContents + ";";
		engine.engEvalString(command);
		
		return this;
	}
	
	public int numCols() {
		return numCols;
	}

	public int numRows() {
		return numRows;
	}

	public void set(int row, int col, double value) {
		if (row < 0 || row >= numRows) {
			throw new IndexOutOfBoundsException("Input row index out of bounds");
		}
		if (col < 0 || col >= numCols) {
			throw new IndexOutOfBoundsException("Input column index out of bounds");
		}
		setQuick(row, col, value);
	}
	
	public void set(double value) {
		String command = matlabMatContents + " = " + value + "* ones(" + numRows
			+ ", " + numCols + ");";
		engine.engEvalString(command);
	}
	
	public void setQuick(int row, int col, double value) {
		String command = matlabMatContents + "(" + (row + 1) + ", " + (col + 1) + ")"
		+ " = " + value + ";";
	engine.engEvalString(command);
	}

	public void setColumnQuick(int col, double[] values) {
		engine.engPutArray("values", values);
		String command = matlabMatContents + "(:, " + (col + 1) + ") = transpose(values);";
		engine.engEvalString(command);
	}
	
	public void setColumn(int col, double[] values) {
	      //		 check size
		if (values.length != numRows) {
			throw new IllegalArgumentException("Size of input array does not match size of Matrix");
		}
		setColumnQuick(col, values);
	}

	public void setRow(int row, double[] values) {
	    //  check size
		if (values.length != numCols) {
			throw new IllegalArgumentException("Size of input array does not match size of Matrix");
		}
		setRowQuick(row, values);
	}
	
	public void setRowQuick(int row, double[] values) {
		engine.engPutArray("values", values);
		String command = matlabMatContents + "(" + (row + 1) + ", :) = values;";
		engine.engEvalString(command);
	}
	
	public void setMatrix(double[][] values) {
		// check size:
//		if (values.length != numRows || values[0].length != numCols) {
//			throw new IllegalArgumentException("Size of input array does not match size of Matrix");
//		}
		
		engine.engPutArray(matlabMatContents, values);
	}
	
	public void setSubMatrix(Matrix subM, int firstRowIdx, int firstColIdx) {
//		if  ((this.numRows < subM.numRows() + firstRowIdx) 
//				||
//			 (this.numCols < subM.numCols() + firstColIdx)) {
//			throw new IllegalArgumentException("Cannot embed given Matrix into "
//					+ "this Matrix at given location: incompatible sizes");
//		}
		int lastRow = firstRowIdx + subM.numRows();
		int lastCol = firstColIdx + subM.numCols();
		String command = matlabMatContents + "(" + (firstRowIdx + 1) + ":"
			+ lastRow + "," + (firstColIdx + 1) + ":" + lastCol + ") = "
			+ ((MatlabMatrix)subM).matlabMatContents + ";";
		engine.engEvalString(command);
	}
	
	public void setSubMatrix(double[][] values, int firstRowIdx, int firstColIdx) {
//		if  ((this.numRows < subM.numRows() + firstRowIdx) 
//				||
//			 (this.numCols < subM.numCols() + firstColIdx)) {
//			throw new IllegalArgumentException("Cannot embed given Matrix into "
//					+ "this Matrix at given location: incompatible sizes");
//		}
		int lastRow = firstRowIdx + values.length;
		int lastCol = firstColIdx + values[0].length;
		engine.engPutArray("subvals", values);
		String command = matlabMatContents + "(" + (firstRowIdx + 1) + ":"
			+ lastRow + "," + (firstColIdx + 1) + ":" + lastCol + ") = subvals;";
		engine.engEvalString(command);
	}
	
	public double[][] toArray() {
		return engine.engGetArray(matlabMatContents);
	}
	
	public double[] toRowPacked1DArray() {
		int matSize = numRows * numCols;
		String command = "contents1D = reshape(transpose(" + matlabMatContents + "), 1, " + matSize + ");";
		engine.engEvalString(command);
		
		double[][] contents2D =  engine.engGetArray("contents1D");
		return contents2D[0];
	}

	public MatlabMatrix transpose() {
	    MatlabMatrix transpose = new MatlabMatrix(numCols, numRows);
	    String command = transpose.matlabMatContents + " = " + matlabMatContents + "';";
	    engine.engEvalString(command);
	    return transpose;
	}
	
	public void setRandom() {
		String command = matlabMatContents + " = rand(" + numRows + ", " + numCols + ");";
		engine.engEvalString(command);
	}
	
	public Matrix sspByRow() {
		MatlabMatrix sspByRowMat = new MatlabMatrix(numRows, numRows);
		String command = sspByRowMat.matlabMatContents + " = " + matlabMatContents 
			+ " * " + matlabMatContents + "';";
		engine.engEvalString(command);
		return sspByRowMat;
	}
	
	public Matrix sspByCol() {
		MatlabMatrix sspByColMat = new MatlabMatrix(numCols, numCols);
		String command = sspByColMat.matlabMatContents + " = " + matlabMatContents 
		    + "' * " + matlabMatContents + ";";
		engine.engEvalString(command);
		return sspByColMat;		
	}
	
	public Matrix copy() {
		return new MatlabMatrix(this.toArray());
	}
	
//	public Matrix subMatrix2(int[] rowRange, int[] colRange) {
////		TODO: can the following error code be incorporated into generic
//		// Matrix class?
//		if ((rowRange.length != 2) || (colRange.length != 2)) {
//			throw new IllegalArgumentException("Input args must be 2-element arrays");
//		}
//		if ((rowRange[1] < rowRange[0]) || (colRange[1] < colRange[0])) {
//			throw new IllegalArgumentException("Invalid row or col range");
//		}
//		if ((rowRange[0] < 0) || (rowRange[1] > this.numRows())
//				|| (colRange[0] < 0) || (colRange[1] > this.numCols())) {
//			throw new IllegalArgumentException("Row/col ranges must be subset of "
//					+ "original Matrix");
//		}
//		
//		int nRows = rowRange[1] - rowRange[0] + 1;
//		int nCols = colRange[1] - colRange[0] + 1;
//		MatlabMatrix subMat = new MatlabMatrix(nRows, nCols, engine);
//		
//		for (int row = 0; row < nRows; ++row) {
//			for (int col = 0; col < nCols; ++col) {
//				subMat.set(row, col, this.get(rowRange[0] + row, colRange[0] + col));
//			}
//		}
//
//		return subMat;	
//	}
	
	public Matrix subMatrix(int[] rowRange, int[] colRange) {
		int nRows = rowRange[1] - rowRange[0] + 1;
		int nCols = colRange[1] - colRange[0] + 1;
		MatlabMatrix subMat = new MatlabMatrix(nRows, nCols);
		
		String command = subMat.matlabMatContents + " = " + matlabMatContents + "(" 
			+ (rowRange[0] + 1) + ":" + (rowRange[1] + 1) + ","
			+ (colRange[0] + 1) + ":" + (colRange[1] + 1) + ");";
		engine.engEvalString(command);
		return subMat;
	}
	
	public Matrix subMatrixRows(int[] rowIndices) {
		MatlabMatrix subMat = new MatlabMatrix(rowIndices.length, numCols);
		double[] rowIndsD = new double[rowIndices.length];
		for (int i = 0; i < rowIndices.length; ++i) {
			rowIndsD[i] = rowIndices[i] + 1;
		}
		String rowIndsML =  subMat.matlabMatContents + "row_inds_submat";
		engine.engPutArray(rowIndsML, rowIndsD);
		String command = subMat.matlabMatContents + " = " + matlabMatContents + 
			"(" + rowIndsML + ",:);";
		engine.engEvalString(command);
		engine.engEvalString("clear " + rowIndsML);
		return subMat;
	}
	
	public Matrix subMatrixCols(int[] colIndices) {
		MatlabMatrix subMat = new MatlabMatrix(numRows, colIndices.length);
		double[] colIndsD = new double[colIndices.length];
		for (int i = 0; i < colIndices.length; ++i) {
			colIndsD[i] = colIndices[i] + 1;
		}
		String colIndsML = subMat.matlabMatContents + "col_inds_submat";
		engine.engPutArray(colIndsML, colIndsD);
		String command = subMat.matlabMatContents + " = " + matlabMatContents + 
			"(:," + colIndsML + ");";
		engine.engEvalString(command);
		engine.engEvalString("clear " + colIndsML);
		return subMat;
	}
	
	 /** Output evects and singular vals are given in DESCENDING order, i.e. dim corresponding to
	 *  largest singular val is first.
	 */
	public npairs.shared.matlib.SingularValueDecomposition svd() {
		// Give each matlab variable its own unique name in case multiple instances of
		// MatlabMatrix call this function concurrently in some way
		// (TODO: investigate synchronization issues)
		String u = "U" + matlabMatContents;
		String v = "V" + matlabMatContents;
		String s = "S" + matlabMatContents;
		String svals = "sVals" + matlabMatContents;
		
		String svdCommand1 = "[" + u + ", " + s + "," + v + "]" +
				" = svd(" + matlabMatContents + ");";
		String svdCommand2 = svals + " = svd(" + matlabMatContents + ");";
		engine.engEvalString(svdCommand1);
		engine.engEvalString(svdCommand2);
		
		double[][] uArray2D = engine.engGetArray(u);
		double[][] vArray2D = engine.engGetArray(v);
		double[][] sArray2D = engine.engGetArray(s);
		double[] sVals1D = trimTo1D(engine.engGetArray(svals));
//		
//		System.out.println("svals: ");
//		npairs.io.NpairsjIO.print(sVals1D);
		
//		int[] sortedIndexOrder = sortDescending(sVals1D);
//		permuteArray(sVals1D, sortedIndexOrder);
//		System.out.println("svals after sorting: ");
//		npairs.io.NpairsjIO.print(sVals1D);
	
		MatlabMatrix uMat = new MatlabMatrix(uArray2D);
//		System.out.println("uMat: ");
//		uMat.print();
//		uMat = (MatlabMatrix)uMat.permuteColumns(sortedIndexOrder);
//		System.out.println("uMat after permutation: ");
//		uMat.print();
//		System.out.println("Perm index array: ");
//		utils_tests.QuickTests.printArray(sortedIndexOrder);
		
		MatlabMatrix vMat = new MatlabMatrix(vArray2D);
//		System.out.println("vMat: ");
//		vMat.print();
//		vMat = (MatlabMatrix)vMat.permuteColumns(sortedIndexOrder);
//		System.out.println("vMat after permutation: ");
//		vMat.print();
		
		MatlabMatrix sMat = new MatlabMatrix(sArray2D);
//		System.out.println("sMat: ");
//		sMat.print();
//		sMat = (MatlabMatrix)sMat.permuteColumns(sortedIndexOrder);
//		System.out.println("sMat after permutation: ");
//		sMat.print();
		
		// clean up matlab variables no longer needed
		engine.engEvalString("clear " + u + " " + 
				v + " " + s + " " + svals);
		
		npairs.shared.matlib.SingularValueDecomposition genericSVD = 
			new npairs.shared.matlib.SingularValueDecomposition(
				uMat, sMat, vMat, sVals1D); 
	
		return genericSVD;
	}
	
	
	/** REQUIRED: Input matrix is square.
	 * 	Output evects and evals are given in DESCENDING order, i.e. dim corresponding to
	 *  largest eval is first.
	 */
	public npairs.shared.matlib.EigenvalueDecomposition eigenvalueDecomposition() {
		// Give each matlab variable its own unique name in case multiple instances of
		// MatlabMatrix call this function concurrently in some way
		// (TODO: investigate synchronization issues)
		String evects = "evects" + matlabMatContents;
		String evals = "evalMat" + matlabMatContents;
		String evdCommand = "[" + evects + ", " + evals + "]" +
				" = eig(" + matlabMatContents + ");";
		engine.engEvalString(evdCommand);
		
		String evalsRe = evals + "Re";
		String evalsIm = evals + "Im";
		// evals are flipped to be in descending order
		String getRealEvalsCommand = evalsRe  + " = real(diag(" + evals + "));";
		String getImagEvalsCommand = evalsIm  + " = imag(diag(" + evals + "));";
		engine.engEvalString(getRealEvalsCommand);
		engine.engEvalString(getImagEvalsCommand);
		
		double[][] evalsRe2D = engine.engGetArray(evalsRe);
		double[] evalsRe1D = trimTo1D(evalsRe2D);
		double[][] evalsIm2D = engine.engGetArray(evalsIm);
		double[] evalsIm1D = trimTo1D(evalsIm2D);
//		
//		System.out.println("Evals before sorting: ");
//		utils_tests.PCATest.printArray(evalsRe1D);
		int[] sortedIndexOrder = sortDescending(evalsRe1D);
		permuteArray(evalsIm1D, sortedIndexOrder);
//		System.out.println("Evals after sorting: ");
//		utils_tests.PCATest.printArray(evalsRe1D);
	
		double[][] evects2D = engine.engGetArray(evects);
		MatlabMatrix eigenVectors = new MatlabMatrix(evects2D);
//		System.out.println("evects before permutation: ");
//		eigenVectors.print();
		eigenVectors = (MatlabMatrix)eigenVectors.permuteColumns(sortedIndexOrder);
//		System.out.println("evects after permutation: ");
//		eigenVectors.print();
//		System.out.println("Perm index array: ");
//		utils_tests.QuickTests.printArray(sortedIndexOrder);
		
		MatlabMatrix realEvalMat = new MatlabMatrix(evalsRe1D.length, evalsRe1D.length);
		try {
			realEvalMat.setDiag(evalsRe1D);
		}
		catch (MatrixException me) {
			// matrix is square by construction
		}
		
		// clean up matlab variables no longer needed
		engine.engEvalString("clear " + evects + " " + 
				evals + " " + evalsRe + " " + evalsIm);
		
		npairs.shared.matlib.EigenvalueDecomposition genericEVD = 
			new npairs.shared.matlib.EigenvalueDecomposition(
				evalsRe1D, 
				evalsIm1D, 
				eigenVectors, 
				getInvSqrtMat(evalsRe1D, engine),
				realEvalMat);
		
		return genericEVD;
	}
	
//	private static double[] trimTo1D(double[][] array2D) {
//		double[] array1D;
//		if (array2D.length == 1) {
//			// array2D is row vector
//			array1D = array2D[0];
//		}
//		else if (array2D[0].length == 1) {
//			// array2D is col vector
//			array1D = new double[array2D.length];
//			for (int i = 0; i < array2D.length; ++i) {
//				array1D[i] = array2D[i][0];
//			}
//		}
//		else throw new IllegalArgumentException
//				("At least one dim of input 2D array must be of length 1");
//		
//		return array1D;
//	}
	
	private static MatlabMatrix getInvSqrtMat(double[] diag_values, JMatLink engine) {
		MatlabMatrix invSqrtMat = new MatlabMatrix(diag_values.length, diag_values.length);
		engine.engPutArray("diag_values", diag_values);
		String command = invSqrtMat.matlabMatContents + " = diag(1./ sqrt(diag_values));";
		engine.engEvalString(command);
		
		// clean up matlab variable no longer needed
		engine.engEvalString("clear diag_values");

		return invSqrtMat;
	}

    // added by Grigori, Feb 25, 2009
    // returns the matrix determinant
    public double det() {
            double d;
            String command = "d = det(" + matlabMatContents + ");";
            engine.engEvalString(command);
            d = engine.engGetScalar("d");
            return d;
    }


	
	public Matrix inverse() {
		MatlabMatrix inverseMat = new MatlabMatrix(numCols, numRows);
		String command;
		if (numRows == numCols) {
			// Matrix is square; use matlab 'inv' function:
			command = inverseMat.matlabMatContents + " = inv(" + matlabMatContents + ");";
		}
		else {
			// Matrix is not square; use matlab 'pinv' function to generate pseudoinverse:
			command = inverseMat.matlabMatContents + " = pinv(" + matlabMatContents + ");";
		}
		engine.engEvalString(command);
		return inverseMat;
	}
	
	public void setIdentity() {
		String command = matlabMatContents + " = eye(" + numRows + ", " + numCols + ");";
		engine.engEvalString(command);
	}
	
    /** Returns new Matrix containing permutation of columns of this 
     * MatlabMatrix according to input index array
     * (e.g. if MatlabMatrix has 3 columns and index array is [1,0,2], then swap
     * first 2 columns of matrix.
     * REQUIRED: input index array has length == no. of cols in matrix.
     * 
     * @param colIndexOrder
     * @return Matrix with columns permuted according to colIndexOrder
     */
    public Matrix permuteColumns(int[] colIndexOrder) {
    	MatlabMatrix permutedMat = new MatlabMatrix(numRows, numCols);
    	for (int i = 0; i < numCols; ++i) {
    		permutedMat.setColumn(i, this.getColumn(colIndexOrder[i]));
    	}
    	return permutedMat;
    }
    
    
	/** Mean-centres columns and divides each column by its standard deviation
	 *  NOTE: this Matrix is modified: columns are mean-centred.
	 * @return normalized column means of this Matrix - each mean divided by column std devs
	 * TODO: add MatlabMatrix.zScoreCols() to take advantage of any Matlab 
	 *       functions for z-score, stddev, etc.
	 */
	public double[] zScoreCols() {
		// NOTE: Matlab 7.4 zscore function returns std and mean of cols, but should
		// not assume that users have 7.4.
		String zsCommand = matlabMatContents + " = zscore(" + matlabMatContents + ");";
		engine.engEvalString(zsCommand);
		String normMeansCommand = matlabMatContents + "normM" + 
				" = mean(" + matlabMatContents + ", 1) ./ std(" + matlabMatContents + ");";
		engine.engEvalString(normMeansCommand);
		double[][] normMeans2D = engine.engGetArray(matlabMatContents + "normM");
		engine.engEvalString("clear " + matlabMatContents + "normM");
		
		return trimTo1D(normMeans2D);
		
	}
	
	
	/** Given input number of rows r and array of row indices, return 
	 *  new Matrix with r rows: rows of this Matrix are contained in 
	 *  rows indicated by input row indices and remaining rows are set 
	 *  to zero.
	 *  REQUIRED: array length == number of rows in this Matrix and
	 *  	r >= number of rows in this Matrix
	 *  @param int num rows in new Matrix 
	 *  @param int array of indices (0-relative) indicating which
	 *         rows of returned Matrix will contain input Matrix rows
	 *  @return Matrix with zero-padded rows
	 */
	public Matrix zeroPadRows(int nRows, int[] rowInds){
		if (nRows < this.numRows) {
			throw new IllegalArgumentException("Input no. of rows is too small.");
		}
		if (rowInds.length != this.numRows) {
			throw new IllegalArgumentException("Input array of row indices is not the right length.");
		}
		
		MatlabMatrix paddedMat = new MatlabMatrix(nRows, this.numCols);
		double[] rowIndsD = new double[numRows];
		for (int i = 0; i < numRows; ++i) {
			rowIndsD[i] = rowInds[i] + 1;
		}
		String rowIndsML = paddedMat.matlabMatContents + "row_inds";
		engine.engPutArray(rowIndsML, rowIndsD);
		String command = paddedMat.matlabMatContents + "(" + rowIndsML + ",:) = " +
			this.matlabMatContents + ";";
		engine.engEvalString(command);
		return paddedMat;
	}

	public static double[] trimTo1D(double[][] array2D) {
		double[] array1D;
		if (array2D.length == 1) {
			// array2D is row vector
			array1D = array2D[0];
		}
		else if (array2D[0].length == 1) {
			// array2D is col vector
			array1D = new double[array2D.length];
			for (int i = 0; i < array2D.length; ++i) {
				array1D[i] = array2D[i][0];
			}
		}
		else throw new IllegalArgumentException
				("At least one dim of input 2D array must be of length 1");
		
		return array1D;
	} 
	
	
	/** Returns new MatlabMatrix contained squared elements of this Matrix.
	 *  Squaring is done element-by-element.
	 *  TODO: test this function
	 *  @return MatlabMatrix containing this MatlabMatrix squared (element-by-element)
	 */
	public MatlabMatrix squareElements() {
		MatlabMatrix squaredMat = new MatlabMatrix(this.numRows, this.numCols);
		String command = squaredMat.matlabMatContents + " = " + this.matlabMatContents 
			+ ".^2;";
		engine.engEvalString(command);
		return squaredMat;
	}
	
	
	protected void finalize() throws Throwable {
		// clean up matlab variable(s) associated with current MatlabMatrix object
		engine.engEvalString("clear " + matlabMatContents);
		if (debug) {
			System.out.println("Freeing Matrix memory (Matlab var " + matlabMatContents + ")...");
			//System.out.println("matlabMatCount = " + matlabMatCount);
		}
		// Don't decrement - otherwise IDs might not be unique, since we
		// don't know how to control e
//		matlabMatCount -= 1;
//		if (matlabMatCount == 0) {
//			if (debug) {
//				System.out.println("Closing matlab engine... ");
//			}
//			engine.engClose();
//			engine = null;
//		}
	}
	
	
//	public static void main(String[] args) {
//		MatlabMatrix M = new MatlabMatrix(5, 4);
//		M.setRandom();
//		System.out.println("Matrix: ");
//		M.print();
//		int[] rowRange = {1,3};
//		int[] colRange = {0,2};
//		System.out.println("Row range: [" + rowRange[0] + "," + rowRange[1] + "]");
//		System.out.println("Col range: [" + colRange[0] + "," + colRange[1] + "]");
//		MatlabMatrix subM = (MatlabMatrix)M.subMatrix(rowRange, colRange);
//		System.out.println("Submatrix: ");
//		subM.print();
//	}
	
	
}
