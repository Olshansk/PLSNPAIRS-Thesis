package npairs.shared.matlib;

import java.io.*;

import npairs.utils.FastQuickSort;

/**
 * Abstract base class for 2D matrices.   
 * @author anita
 *
 */
public abstract class Matrix {

	/**
	 * Returns number of rows in this Matrix.
	 * 
	 * @return number of rows
	 */
	abstract public int numRows();

	/**
	 * Returns number of columns in this Matrix.
	 * 
	 * @return number of columns
	 */
	abstract public int numCols();

	/**
	 * Sets contents of this Matrix to values.
	 * 
	 * @param values
	 *            2D array of doubles: values[i][j] == element at ith row, jth
	 *            col.
	 */
	abstract public void setMatrix(double[][] values);

	/** 
	 * Sets subset of this Matrix to values in subM.
	 * @param subM
	 * 			Matrix smaller than or equal in size to this Matrix 
	 * @param firstRowIdx
	 * 			row index of this Matrix to set to subM(0,0)
	 * @param firstColIdx
	 * 			col index of this Matrix to set to subM(0,0)
	 * @exception IllegalArgumentException 
	 *       	if subM is larger than this Matrix 
	 *       or if firstRowIdx or firstColIdx are < 0 or too large to allow subM 
	 *          to fit fully into this Matrix 
	 *                                     
	 */
	abstract public void setSubMatrix(Matrix subM, int firstRowIdx, int firstColIdx);
	
	
	/** 
	 * Sets subset of this Matrix to values.
	 * @param values
	 * 			2D array of doubles smaller than or equal in size to this Matrix 
	 * @param firstRowIdx
	 * 			row index of this Matrix to set to values[0][0]
	 * @param firstColIdx
	 * 			col index of this Matrix to set to values[0][0]
	 * @exception IllegalArgumentException 
	 *       	if values array is larger than this Matrix 
	 *       or if firstRowIdx or firstColIdx are < 0 or too large to allow values 
	 *          to fit fully into this Matrix 
	 *                                     
	 */
	abstract public void setSubMatrix(double[][] values, int firstRowIdx, int firstColIdx);
	
	
	/**
	 * Returns array containing values in given row of this Matrix.
	 * 
	 * @param row 
	 *            Which row to return (0-relative)
	 * @return Array of values in given row
	 */
	abstract public double[] getRow(int row);

	abstract public double[] getRowQuick(int row);
	/**
	 * Sets given row of this Matrix to given values.
	 * 
	 * @param row 
	 *            Which row to set (0-relative)
	 * @param values 
	 *            Values to put in given row
	 */
	abstract public void setRow(int row, double[] values);

	abstract public void setRowQuick(int row, double[] values);
	
	/**
	 * Returns array containing values in given column of this Matrix.
	 * 
	 * @param col 
	 *            Which col to return (0-relative)
	 * @return Array of values in given col
	 */
	abstract public double[] getColumn(int col);

	abstract public double[] getColumnQuick(int col);
	
	/** Sets given column of this Matrix to given values.
	 * 
	 * @param col
	 * 			  Which column to set (0-relative)
	 * @param values
	 * 		      Values to put in given column
	 */
	abstract public void setColumn(int col, double[] values);
		
	abstract public void setColumnQuick(int col, double[] values);
	
	/** Gets element at given location (0-relative) in this Matrix
	 * 
	 * @param row
	 * @param col
	 * @see #set(int, int, double)
	 * @see #getQuick(int, int)
	 * @return Value at (row, col) 
	 */
	abstract public double get(int row, int col);

	/** Gets element at given location (0-relative) in this Matrix
	 * 	without checking whether input indices are out of bounds.
	 * 
	 * @param row
	 * @param col
	 * @see #set(int, int, double)
	 * @see #get(int, int)
	 * @return Value at (row, col) 
	 */
	abstract public double getQuick(int row, int col);
	
	/** Sets element at given location (0-relative) in this Matrix
	 *  to given value
	 * 
	 * @param row
	 * @param col
	 * @param value
	 * @see #get(int, int)
	 */
	abstract public void set(int row, int col, double value);
	
	
	/**
	 * Sets all elements in this Matrix to given value.
	 * 
	 * @param value
	 */
	abstract public void set(double value);
	

	abstract public void setQuick(int row, int col, double value);
	
	/** Fills this Matrix with 1s on the main diagonal and 0s everywhere else.
	 */
	abstract public void setIdentity();

	/** Sets diagonal values in this Matrix to input diagonal values.
	 *  This Matrix must be square of size N X N, where N = length of
	 *  diagVals.
	 *  
	 * @param diagVals
	 * @return this Matrix 
	 */
	public void setDiag(double[] diagVals) throws MatrixException {
		if (!this.isSquare()) {
			throw new MatrixException("Matrix must be square.");
		}
		if (diagVals.length != this.numRows()) { 
			throw new IllegalArgumentException("Length of input array of diagonal " +
			"elements must be equal to no. of rows and columns in Matrix.");
		}
		for (int i = 0; i < this.numRows(); ++i) {
			this.setQuick(i, i, diagVals[i]);
		}
	}
	
	
	/** 
	 * Mean-centres columns of this Matrix.  This Matrix is modified.
	 * @return array of column means
	 */
	abstract public double[] meanCentreColumnsInPlace();
	
	/** Mean-centres columns of this Matrix. This Matrix is NOT modified.
	 */
	abstract public Matrix meanCentreColumns();
	
	/** Returns contents of this Matrix as 2D array of doubles.	 * 
	 * @return 2D array containing values in this Matrix.
	 */
	abstract public double[][] toArray();

	/** Returns contents of this Matrix as 1D array of doubles, one row 
	 *  at a time, from row 0 to last row.
	 *  
	 *  @return 1D array containing values in this Matrix, one row at a time
	 */
	abstract public double[] toRowPacked1DArray();
	
	/**
	 * Constructs and returns new Matrix containing result when this Matrix is 
	 * matrix-multiplied on the right by input Matrix.
	 * I.e., if this Matrix == M, then result == MB.
	 * 
	 * REQUIRED: argument belongs to same concrete Matrix class as 'this'.
	 * @param B 
	 * 			Matrix to be right-multiplied with this Matrix 
	 * @return Matrix multiplication MB (where M == this Matrix)
	 * @see #mult(double)
	 */
	abstract public Matrix mult(Matrix B);

	/**
	 * Constructs and returns new Matrix containing result when this Matrix is 
	 * scalar-multiplied by 'scalar'.
	 * 
	 * @param scalar 
	 * 			Factor by which to multiply every element in this Matrix
	 * @return scalar * this Matrix
	 * @see #mult(Matrix)
	 */
	abstract public Matrix mult(double scalar);
	
	/** Multiplies every element in this Matrix by 'scalar'.  
	 *  This Matrix is modified.  Result is also returned for
	 *  convenience.
	 */
	abstract public Matrix multInPlace(double scalar);

	
	/**
	 * Adds Matrix B to this Matrix, element-by-element, and stores the result
	 * in this Matrix. This Matrix is modified.
	 * REQUIRED: argument belongs to same concrete Matrix class as
	 * 'this'.
	 * 
	 * @param B
	 *            Matrix to be added to this Matrix.  This Matrix is modified.
	 */
	// TODO: get plusEquals() to return this (modified) Matrix so it's compatible
	// with minusEquals() below.
	abstract public Matrix plusEquals(Matrix B);
	
	/** Adds Matrix B to this Matrix, element-by-element, and returns a new 
	 *  Matrix containing the result.  
	 *  REQUIRED: argument belongs to same concrete Matrix class as 
	 *  this Matrix.
	 *   
	 * @param B
	 *            Matrix to be added to this Matrix.  This Matrix is not 
	 *            modified.
	 */
	 public Matrix plus(Matrix B) {
		 Matrix A = this.copy();
		 A = A.plusEquals(B);
		 return A;
	 }
	 
	 /** Adds scalar to each element in this Matrix.  
	  * This Matrix is modified.
	  *
	  * @param scalar 
	  */
	 public void plusEquals(double scalar) {
		 for (int c = 0; c < this.numCols(); ++c) {
			 for (int r = 0; r < this.numRows(); ++r) {
				 this.set(r, c, this.get(r, c) + scalar);
			 }
		}
	 }
	 
	
	 /** Subtracts Matrix B from this Matrix, element-by-element, and stores the
	  * result in this Matrix.  This Matrix is modified.
	  *  REQUIRED: argument belongs to same concrete Matrix class as 
	  *  this Matrix.	  
	  *    
	  * @param B
	  *            Matrix to be added to this Matrix.  This Matrix is
	  *            modified.
	  */
	 abstract public Matrix minusEquals(Matrix B);
	 
	 /** Subtracts Matrix B from this Matrix, element-by-element, and returns a new 
		 *  Matrix containing the result.  
		 *  REQUIRED: argument belongs to same concrete Matrix class as 
		 *  this Matrix.
		 *   
		 * @param B
		 *            Matrix to be added to this Matrix.  This Matrix is not 
		 *            modified.
		 */
		 public Matrix minus(Matrix B) {
			 Matrix A = this.copy();
			 A = A.minusEquals(B);
			 return A;
		 }

	/**
	 * Constructs and returns new Matrix containing result of matrix
	 * multiplication M * M' (where M is this Matrix, and M' == transpose of M).
	 * (SSP stands for 'sums of squares and products').
	 * M is NOT mean-centred before multiplication.
	 * 
	 * @return M * M', where M is this Matrix, * is matrix multiplication and M'
	 *         is the transpose of M
	 * @see #sspByCol()
	 */
	abstract public Matrix sspByRow();

	/**
	 * Constructs and returns new Matrix containing result of matrix
	 * multiplication M' * M (where M is this Matrix, and M' == transpose of M).
	 * (SSP stands for 'sums of squares and products').
	 * M is NOT mean-centred
	 * before multiplication.
	 * 
	 * @return M' * M, where M is this Matrix, * is matrix multiplication and M'
	 *         is the transpose of M
	 * @see #sspByRow()
	 */
	abstract public Matrix sspByCol();

	/** Constructs and returns new Matrix containing transpose of this Matrix.
	 *  Note however that for all Colt Matrix libraries (Parallel Colt + regular
	 *  Colt), underlying DenseDoubleMatrix2D is the same as original one, not a copy,
	 *  so changes to this Matrix will be reflected in transposed Matrix and vice
	 *  versa.  This is NOT the case with MatlabMatrix, where underlying data structure
	 *  (matlab array) is not the same, so changes in original Matrix will NOT be reflected
	 *  in transposed Matrix or vice versa.  
	 * 
	 * @return transpose of this Matrix
	 */
	abstract public Matrix transpose();

	/**
	 * Sets this Matrix to random double values between 0 and 1. 
	 */
	abstract public void setRandom();

	/** Mean-centres columns and divides each column by its standard deviation
	 *  NOTE: this Matrix is modified: columns are mean-centred.
	 * @return column means of this Matrix - each mean divided by column std devs
	 * NOTE: overridden by MatlabMatrix but not ColtMatrix
	 */
	public double[] zScoreCols() {
		
		int numRows = this.numRows();
		int numCols = this.numCols();
		double[] normedMeans = this.meanCentreColumnsInPlace();  
		for (int c = 0; c < numCols; ++c) {
			double var = 0.0;
			for (int r = 0; r < numRows; ++r) {
				var += Math.pow(this.get(r, c), 2.0);
			}
			//System.out.println("Var before division by numRows: " + var);
			var /= (numRows - 1);
			//System.out.println("Var: " + var);
			double stdDev = Math.sqrt(var);
			for (int r = 0; r < numRows; ++r) {
				
				this.setQuick(r, c, (this.getQuick(r,c) / stdDev));
			}
			normedMeans[c] /= stdDev;

		}
		//System.out.println("Normed means (Matrix.zScoreCols(double[]):");
//		npairs.io.NpairsjIO.print(normedMeans);
		return normedMeans;
	}
	
	/** Given input number of rows r and array of row indices, return 
	 *  new Matrix with r rows - rows of this Matrix are contained in 
	 *  rows indicated by input row indices and remaining rows are set 
	 *  to zero.
	 *  REQUIRED: array length == number of rows in this Matrix and
	 *  	r >= number of rows in this Matrix
	 *  @param int num rows in new Matrix 
	 *  @param int array of indices (0-relative) indicating which
	 *         rows of returned Matrix will contain input Matrix rows
	 *  @return Matrix with zero-padded rows
	 */
	abstract public Matrix zeroPadRows(int nRows, int[] rowInds);

	
	/**
	 * Performs eigenvalue decomposition of this Matrix.
	 * REQUIRED: this Matrix is square.
	 * 
	 * @return EigenvalueDecomposition object containing eigenvectors and 
	 * 		   eigenvalues of this Matrix. Evects and evals are in 
	 * 		   descending order, from largest (real) eval to smallest.
	 * @see matlib.EigenvalueDecomposition
	 */
	abstract public EigenvalueDecomposition eigenvalueDecomposition();

	/** Performs singular value decomposition of this Matrix.
	 *  REQUIRED: this Matrix numRows >= numCols
	 *  
	 * @return SingularValueDecomposition object containing results from
	 *         singular value decomposition 
	 *         svd(thisMatrix) = U * S * transpose(V).
	 * @see matlib.SingularValueDecomposition      
	 * 
	 */
	abstract public SingularValueDecomposition svd();
	/**
	 *  Prints contents of this Matrix to standard output stream.
	 *  @see #printToFile(String, String)
	 */
	public void print() {
		double[][] matContents = this.toArray();
		for (int row = 0; row < this.numRows(); ++row) {
			for (int col = 0; col < this.numCols(); ++col) {
				System.out.print(matContents[row][col] + " ");
			}
			System.out.println();
		}
	}
	/**
	 *  Prints contents of this Matrix to standard output stream.
	 *  Prints 3 digits after decimal place.
	 *  @see #printToFile(String, String)
	 */
	public void printf() {
		double[][] matContents = this.toArray();
		for (int row = 0; row < this.numRows(); ++row) {
			for (int col = 0; col < this.numCols(); ++col) {
				System.out.printf("%.3f ", matContents[row][col]);
			}
			System.out.println();
		}
	}
	
//	/**
//	 * Saves contents of this Matrix to file.
//	 * 
//	 * @param fname 
//	 * 			Name of file in which to save this Matrix. 
//	 * @param format
//	 * 			Format to use when saving this Matrix.
//	 * 			(Currently, only valid formatting options are 'IDL',
//	 * 			 which formats Matrix so that it can be read by 
//	 * 			Jon Anderson's 'read_matrix.pro' procedure; and 
//	 * 			"default", which prints Matrix such that rows
//	 *          of textfile correspond to rows of Matrix.  
//	 * 			Override to add more formatting options.)
//	 * @see #print()
//	 */
//	public void printToFile(String fname, String format) {
//		double[][] matContents = this.toArray();
//		FileOutputStream fos = null;
//		PrintStream ps;
//		
//		if (format.toUpperCase().equals("DEFAULT")) {
//			try {
//				fos = new FileOutputStream(fname);
//				ps = new PrintStream(fos);
//				for (int row = 0; row < this.numRows(); ++row) {
//					for (int col = 0; col < this.numCols(); ++col) {
//						ps.print(matContents[row][col] + " ");
//					}
//					ps.println();
//				}
//			}
//			catch (IOException e) {
//				System.err.println("Error writing Matrix to file " + fname);
//				e.printStackTrace();
//			}
//			finally {
//				try {
//					if (fos != null) {
//						fos.close();
//					}
//				}
//				catch (Exception e) {
//					System.err.println("Error - could not close file " + fname);
//					e.printStackTrace();
//				}
//			}
//		}
//		
//		else if (format.toUpperCase().equals("IDL")) {
//			try {
//				fos = new FileOutputStream(fname);
//				ps = new PrintStream(fos);
//				ps.println(this.numCols() + " " + this.numRows());
//				for (int row = 0; row < this.numRows(); ++row) {
//					for (int col = 0; col < this.numCols(); ++col) {
//						ps.print(matContents[row][col] + " ");
//					}
//					ps.println();
//				}
//			} catch (Exception e) {
//				System.err.println("Error writing Matrix to file" + fname);
//			}
//			finally {
//				try {
//					if (fos != null) {
//						fos.close();
//					}
//				}
//				catch (Exception e) {
//					System.err.println("Error - could not close file " + fname);
//					e.printStackTrace();
//				}	
//			}
//		}
//	}
//	

	/**
	 * Saves contents of this Matrix to file.
	 * 
	 * @param fname 
	 * 			Name of file in which to save this Matrix. 
	 * @param format
	 * 			Format to use when saving this Matrix.
	 * 			(Currently, only valid formatting options are 'IDL',
	 * 			 which formats Matrix so that it can be read by 
	 * 			Jon Anderson's 'read_matrix.pro' procedure; and 
	 * 			"default", which prints Matrix such that rows
	 *          of textfile correspond to rows of Matrix.  
	 * 			Override to add more formatting options.)
	 * @see #print()
	 */
	public void printToFile(String fname, String format) {
		double[][] matContents = this.toArray();
		PrintWriter pw = null;
		
		if (format.toUpperCase().equals("DEFAULT")) {
			try {
				pw = new PrintWriter(new BufferedWriter(new FileWriter(fname)));
				
				for (int row = 0; row < this.numRows(); ++row) {
					for (int col = 0; col < this.numCols(); ++col) {
						pw.print(matContents[row][col] + " ");
					}
					pw.println();
				}
			}
			catch (IOException e) {
				System.err.println("Error writing Matrix to file " + fname);
				e.printStackTrace();
			}
			finally {
				try {
					if (pw != null) {
						boolean error = pw.checkError();
						if (error) {
							throw new IOException("Error occurred writing to file "
									+ fname);
						}
						pw.close();
					}
				}

				catch (IOException e) {
					e.printStackTrace();
				}
				catch (Exception e) {
					System.err.println("Error - could not close file " + fname);
					e.printStackTrace();
				}
			}
		}
		
		else if (format.toUpperCase().equals("IDL")) {
			try {
				pw = new PrintWriter(new BufferedWriter(new FileWriter(fname)));
				
				pw.println(this.numCols() + " " + this.numRows());
				for (int row = 0; row < this.numRows(); ++row) {
					for (int col = 0; col < this.numCols(); ++col) {
						pw.print(matContents[row][col] + " ");
					}
					pw.println();
				}
			} 
			catch (Exception e) {
				System.err.println("Error writing Matrix to file" + fname);
			}
			finally {
				try {
					if (pw != null) {
						boolean error = pw.checkError();
						if (error) {
							throw new IOException("Error occurred writing to file "
									+ fname);
						}
						pw.close();
					}
				}
				catch (IOException e) {
					e.printStackTrace();
				}
				catch (Exception e) {
					System.err.println("Error - could not close file " + fname);
					e.printStackTrace();
				}
			}
		}
	}
	
	/**
	 * Constructs and returns a deep copy of this Matrix.
	 * 
	 * @return deep copy of this Matrix.
	 */
	abstract public Matrix copy();

	
	/**
	 * Returns true if elements of Matrices M1 and M2 are equal, to within
	 * equalityThresh. Otherwise returns false.
	 * 
	 * @param M1 First Matrix
	 * @param M2 Second Matrix
	 * @param equalityThresh
	 * 			Threshold of equality.  M1 and M2 must differ by less than 
	 * 			equalityThresh at every location to be considered equal.
	 * @return true if Matrices are equal; false otherwise.
	 */
	// TODO: optimize Matlab equal(Mat,Mat): just compare MatlabMatrix contents
	// directly in Matlab instead
	// of going via Matrix.get(int,int)
	public static boolean equal(Matrix M1, Matrix M2, double equalityThresh) {
		boolean equal = true;
		if ((M1.numRows() != M2.numRows()) || (M1.numCols() != M2.numCols())) {
			equal = false;
		} else {
			for (int row = 0; row < M1.numRows(); ++row) {
				for (int col = 0; col < M1.numCols(); ++col) {
					if (Math.abs(M1.get(row, col) - M2.get(row, col)) > equalityThresh) {
						equal = false;
						break;
					}
				}
				if (!equal) {
					break;
				}
			}
		}
		return equal;
	}

	// TODO: should move static methods equal(double[], double[], double) and
	// scale(double[], double) out of Matrix class.


	/**
	 * Constructs and returns new Matrix containing subMatrix of this Matrix.
	 * 
	 * @param rowRange -
	 *            2-element int array: [firstRowIndex, lastRowIndex]
	 * @param colRange -
	 *            2-element int array: [firstColIndex, lastColIndex]
	 * @return subMatrix of this Matrix as delineated by rowRange, colRange
	 * @see #subMatrixRows(int[])
	 * @see #subMatrixCols(int[])
	 */
	abstract public Matrix subMatrix(int[] rowRange, int[] colRange);

	/**
	 * Constructs and returns new Matrix containing just the rows indicated by
	 * input array of row indices. Size of returned Matrix: length of rowIndices
	 * array X this.numCols()
	 * 
	 * @param rowIndices
	 *            int array of indices of rows of this matrix to be included in
	 *            returned Matrix.
	 * @return Matrix containing just the given rows of this Matrix.
	 * @see #subMatrix(int[], int[])
	 * @see #subMatrixCols(int[])
	 */
	abstract public Matrix subMatrixRows(int[] rowIndices);

	/**
	 * Constructs and returns new Matrix containing just the cols indicated by
	 * input array of col indices. Size of returned Matrix: this.numRows() 
	 * X length of colIndices array
	 * 
	 * @param colIndices
	 *            int array of indices of cols of this matrix to be included in
	 *            returned Matrix.
	 * @return Matrix containing just the given cols of this Matrix.
	 * @see #subMatrix(int[], int[])
	 * @see #subMatrixRows(int[])
	 */
	abstract public Matrix subMatrixCols(int[] colIndices);
	
	
	
    /** Returns the determinant (added by Grigori on Feb 25, 2009)
    *
    * @return determinant
    */

   abstract public double det();



	/**
	 * Constructs and returns new Matrix containing inverse of this Matrix, if
	 * Matrix is square; if Matrix is not square, returns pseudoinverse.
	 * 
	 * @return Inverse or pseudoinverse of this Matrix
	 */
	abstract public Matrix inverse();

	// TODO: check that equality tolerance in matlab and colt inverse code is
	// comparable
	// TODO: check what happens in matlab and colt when matrix is square but
	// badly cond. or
	// singular.

	public boolean isSquare() {
		return (this.numCols()== this.numRows());
	}
	/**
	 * Returns new Matrix containing permutation of columns of this Matrix
	 * according to input index array. E.g., if Matrix has 3 columns and index
	 * array is [1,0,2], then swap first 2 columns of matrix. REQUIRED: input
	 * index array has length == no. of cols in matrix.
	 * 
	 * @param colIndexOrder
	 * 			Order of columns in new Matrix
	 * @return Matrix with columns permuted according to colIndexOrder
	 */
	abstract public Matrix permuteColumns(int[] colIndexOrder);
	
	// TODO: Move the following static methods into MLFuncs 

	/**
	 * Returns a new array containing elements of input array in reverse order.
	 * 
	 * @param values -
	 *            double array
	 * @see #reverse(int[])
	 */
	protected static void reverse(double[] values) {
		double[] tempVals = values.clone();
		for (int i = 0; i < values.length; ++i) {
			values[i] = tempVals[(values.length - 1) - i];
		}
	}

	/**
	 * Returns a new array containing elements of input array in reverse order.
	 * 
	 * @param values -
	 *            int array
	 * @see #reverse(double[])
	 *         
	 */
	protected static void reverse(int[] values) {
		int[] tempVals = values.clone();
		for (int i = 0; i < values.length; ++i) {
			values[i] = tempVals[(values.length - 1) - i];
		}

	}

	/**
	 * Sorts the input array into descending order and returns array of indices
	 * indicating the new ordering. (E.g. {1, 5, -2} would be rearranged into {5,
	 * 1, -2} and returns the array {1, 0, 2})
	 * 
	 */
	protected static int[] sortDescending(double[] values) {
		FastQuickSort f = new FastQuickSort();
		f.sort(values);
		reverse(values);
		int[] indexArray = f.sortedIndex;
		reverse(indexArray);
		return indexArray;
	}

	/**
	 * Reorders input double array into order given by input index array. (E.g.
	 * if input double array is [4,8,5] and index array is [1,0,2] then [4, 8,
	 * 5] becomes [8, 4, 5]) REQUIRED: input arrays are of same length; indices
	 * are permutation of {0, 1, ..., arrayLength-1}
	 */
	protected static void permuteArray(double[] values, int[] indexOrder) {
		double[] tempVals = values.clone();
		for (int i = 0; i < values.length; ++i) {
			values[i] = tempVals[indexOrder[i]];
		}
	}
	
	
	/** Returns true if array1 and array2 are equal, to within equalityThresh.
	 *  Otherwise returns false.
	 *  
	 *  @param array1 
	 *  		First array
	 *  @param array2
	 *  		Second array
	 *  @param equalityThresh 
	 *  		Threshold of equality.  array1 and array2 must differ by less than 
	 *  		equalityThresh at every location to be considered equal.
	 *  @return true if arrays are equal; false otherwise.
	 */
	public static boolean equal(double[] array1, double[] array2, double equalityThresh) {
		boolean equal = true;
		if (array1.length != array2.length) {
			equal = false;
		} else {
			for (int i = 0; i < array1.length; ++i) {
				if (Math.abs(array1[i] - array2[i]) > equalityThresh) {
					equal = false;
					break;
				}
			}
		}
		return equal;
	}
	
	/** Returns array of column means
	 * 
	 * @return double array containing column means of this Matrix
	 */
	abstract public double[] colMeans();
	
	/** Calculates correlation of this Matrix and input Matrix.  
	 * @param Matrix - must have same number of rows as this Matrix.
	 * @return Matrix containing Pearson's correlation coefficients of 
	 *  each column of input Matrix with each column of this Matrix.
	 *  Dimensions of returned Matrix: 
	 *  (no. cols in this Matrix) rows X (no. cols in input Matrix) cols
	 */
	abstract public Matrix correlate(Matrix M);
	
	/** Returns dot product of input arrays. 
	 * @param x
	 * @param y
	 * @return dot product of x and y
	 * @throws IllegalArgumentException if input arrays are not
	 *          the same length.
	 */
	
	/** Returns new Matrix contained squared elements of this Matrix.
	 *  Squaring is done element-by-element.
	 *  @return this Matrix squared (element-by-element)
	 */
	abstract public Matrix squareElements();
	
	protected static double dotProd(double[] x, double[] y) {
		if (x.length != y.length) {
			throw new IllegalArgumentException("Input arrays must be same length.");
		}
		double dProd = 0;
		for (int i = 0; i < x.length; ++i) {
			dProd += x[i]*y[i];
		}
		return dProd;
	}
	
}

