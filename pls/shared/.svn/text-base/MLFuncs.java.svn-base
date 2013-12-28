package pls.shared;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Vector;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;

import npairs.io.NpairsjIO;


import com.jmatio.types.MLCell;
import com.jmatio.types.MLChar;

import extern.ArrayFuncs;
import extern.FastQuickSort;


import Jama.Matrix;

/**
 * A static class that provides some equivalant Matlab functions.  Also some convenience functions which
 * would be simple to code in Matlab are included here.
 * 
 * @author imran
 */
public class MLFuncs {
	
	/**
	 * Generate the Helmert Matrix with a given row size.
	 * @param numRows the number of rows in the Helmert Matrix
	 * @return the Helmert matrix with the given number of rows
	 */
	public static double[][] helmertMatrix(int n) {
		double[][] helmert = new double[n][n];
		for(int i = 1; i <= n; i++) {
			for(int j = 1; j <= n; j++) {
				if(i == 1) {
					helmert[i - 1][j - 1] = 1/Math.sqrt(n);
				} else if(j < i) {
					helmert[i - 1][j - 1] = 1/Math.sqrt(i * (i - 1)); 
				} else if(i == j) {
					helmert[i - 1][j - 1] = (1 - i)/Math.sqrt(i * (i - 1));
				} else {
					helmert[i - 1][j - 1] = 0;
				}
			}
		}
		return helmert;
	}

	public static double[][] getRRIHelmertMatrix(int n) {
		double[][] helmert = MLFuncs.helmertMatrix(n);
		double[][] modified = MLFuncs.flipDiagonal(helmert);
		modified = MLFuncs.times(modified, -1);
		modified = MLFuncs.getRows(modified, MLFuncs.range(0, modified.length - 2));
		return modified;
	}
	
	public static Matrix std(Matrix M) {
		return sqrt(var(M));
	}
	
	/** Returns sample variance of input array of values.
	 * 
	 * @param array of values (doubles)
	 * @return sample variance 
	 */
	public static double var(double[] vector) {
		int n = vector.length;
		
		double sum = 0;
		
		for(double v : vector) {
			sum += v;
		}
		
		double mean = sum / n;
		
		double var = 0;
		
		for(double v : vector) {
			var += Math.pow(v - mean, 2);
		}
		
		return var / (n - 1);
	}
	
	/** Returns sample standard deviation of input array of doubles
	 * 
	 * @param array of values (doubles)
	 * @return sample standard deviation
	 */
	public static double std(double[] vector) {
		return Math.sqrt(var(vector));
	}
	
	/** Returns sample covariance between input double arrays
	 * 
	 * @param values1
	 * @param values2
	 * @return sample covariance between input arrays
	 */
	private static double cov(double[] values1, double[] values2) {
		if (values1.length != values2.length) {
			throw new IllegalArgumentException("Input arrays must be of same length.");
		}
		
		double mean1 = avg(values1);
		double mean2 = avg(values2);
		int numPts = values1.length;
		double cov = 0;
		for (int i = 0; i < numPts; ++i) {
			cov +=  (values1[i] - mean1) * (values2[i] - mean2) / (numPts - 1);
		}
		return cov;	
	}
	
	/** Returns sample correlation (Pearson's r) between input arrays of doubles.
	 * 
	 * @param values1
	 * @param values2
	 * @return Pearson's sample correlation coefficient (r) between values1 & values2
	 */
	public static double corr(double[] values1, double[] values2) {
		double cov = cov(values1, values2);
		double std1 = std(values1);
		double std2 = std(values2);
		double r = cov / (std1 * std2);
		return r;
	}
	
	public static void replaceLessThanOrEqualToZeroWithOnes(Matrix M) {
		for(int i = 0; i < M.getRowDimension(); i++) {
			for(int j = 0; j < M.getColumnDimension(); j++) {
				if(M.get(i, j) <= 0) {
					M.set(i, j, 1);
				}
			}
		}
	}
	
	public static boolean all(boolean[] array) {
		for(boolean b : array) {
			if(!b) return false;
		}
		return true;
	}
	
	public static boolean containsZero(Matrix M) {
		for(int i = 0; i < M.getRowDimension(); i++) {
			for(int j = 0; j < M.getColumnDimension(); j++) {
				if(M.get(i, j) == 0) {
					return true;
				}
			}
		}
		return false;
	}
	
	public static boolean any(Matrix M) {
		for(int i = 0; i < M.getRowDimension(); i++) {
			for(int j = 0; j < M.getColumnDimension(); j++) {
				if(M.get(i, j) != 0) {
					return true;
				}
			}
		}
		return false;
	}
	
	public static Matrix var(Matrix M) {
		int m = M.getRowDimension();
		int n = M.getColumnDimension();
		
		Matrix ret = new Matrix(1, n);
		for(int i = 0; i < n; i++) {
			double[] column = M.getMatrix(0, m - 1, i, i).transpose().getArray()[0];
			double currVar = var(column);
			ret.set(0, i, currVar);
		}
		return ret;
	}
	
	public static double[][] load(String fileName, int numberOfLinesToSkip, String skipLinesBeginningWith) {
		BufferedReader in = null;
		
		try {
			in = new BufferedReader(new InputStreamReader(new FileInputStream(fileName)));
		} catch(Exception ex) {
			return null;
		}
		
		Vector<String[]> allLines = new Vector<String[]>();
		String thisLine = null;
		
		try {
			for(int i = 0; i < numberOfLinesToSkip; i++, in.readLine());
			while((thisLine = in.readLine()) != null) {
				thisLine = thisLine.trim();
				if(!thisLine.equals("") && (skipLinesBeginningWith == null || (skipLinesBeginningWith != null && !thisLine.startsWith(skipLinesBeginningWith)))) {
					allLines.add(thisLine.split("\\s+"));
				}
			}
		} catch(Exception ex) {
			return null;
		}
		
		double ret[][] = new double[allLines.size()][allLines.get(0).length];
		
		for(int i = 0; i < ret.length; i++) {
			for(int j = 0; j < ret[0].length; j++) {
				ret[i][j] = new Double(allLines.get(i)[j]).doubleValue();
			}
		}
		
		return ret;
	}

	public static double[][] load(String fileName) {
		return load(fileName, 0, null);
	}

	public static double[][] load(String fileName, int numberOfLinesToSkip) {
		return load(fileName, numberOfLinesToSkip, null);
	}

	public static double[][] load(String fileName, String skipLinesBeginningWith) {
		return load(fileName, 0, skipLinesBeginningWith);
	}
	
	public static double[][] flipDiagonal(double[][] data) {
		int m = data.length;
		int n = data[0].length;
		double[][] ret = new double[m][n];
		for(int i = 0; i < m; i++) {
			for(int j = 0; j < n; j++) {
				ret[m - i - 1][n - j - 1] = data[i][j];
			}
		}
		return ret;
	}
	
	public static int[] setValues(int[] data, int[] indeces, int value) {
		int[] ret = data.clone();
		for(int i : indeces) {
			ret[i] = value;
		}
		return ret;
	}
	
	public static double[] setValues(double[] data, int[] indeces, double value) {
		double[] ret = data.clone();
		for(int i : indeces) {
			ret[i] = value;
		}
		return ret;
	}
	
	public static Matrix setValues(Matrix data, int[] indeces, double value) {
		int m = data.getRowDimension();
		return new Matrix(setValues(flattenVertically(data), indeces, value), m);
	}
	
	public static Matrix ceil(Matrix data) {
		int m = data.getRowDimension();
		int n = data.getColumnDimension();
		Matrix ret = data.copy();
		for(int i = 0; i < m ; i++) {
			for(int j = 0; j < n; j++) {
				ret.set(i, j, Math.ceil(ret.get(i, j)));
			}
		}
		return ret;
	}
	
	public static double[][] toDoubleArray(int[] data) {
		double[][] ret = new double[1][data.length];
		ret[0] = (double[])ArrayFuncs.convertArray(data, double.class);
		return ret;
	}
	
	public static double[][] toDoubleArray(int[][] data) {
		double[][] ret = (double[][])ArrayFuncs.convertArray(data, double.class);
		return ret;
	}
	
	/** Turns 1D array into "2D" array (with one "row")
	 * 
	 * @param data
	 * @return data as (trivially) 2D array
	 */
	public static double[][] to2DArray(double[] data) {
		double[][] ret = new double[1][data.length];
		ret[0] = data;
		return ret;
	}
	
	public static boolean isEqual(int[] a, int[] b) {
		return Arrays.equals(a, b);
	}
	
	public static boolean isEqual(String[] a, String[] b) {
		return Arrays.equals(a, b);
	}
	
	public static int[][] appendColumn(int[][] data, int[][] appendage) {
		if(data == null) {
			return appendage;
		}
		int[][] ret = new int[data.length][data[0].length + appendage[0].length];
		for(int i = 0; i < data.length; i++) {
			for(int j = 0; j < data[0].length; j++) {
				ret[i][j] = data[i][j];
			}
		}
		for(int i = 0; i < appendage.length; i++) {
			for(int j = 0; j < appendage[0].length; j++) {
				ret[i][j + data[0].length] = appendage[i][j];
			}
		}
		return ret;
	}
	
	public static int[] append(int[] data, int[] appendage) {
		if(data == null) {
			return appendage;
		}
		int[] ret = new int[data.length + appendage.length];
		for(int i = 0; i < data.length; i++) {
			ret[i] = data[i];
		}
		for(int i = 0; i < appendage.length; i++) {
			ret[i + data.length] = appendage[i];
		}
		return ret;
	}
	
	public static int[][] append(int[][] data, int[][] appendage) {
		if(data == null) {
			return appendage;
		}
		int[][] ret = new int[data.length + appendage.length][data[0].length];
		for(int i = 0; i < data.length; i++) {
			ret[i] = data[i];
		}
		for(int i = 0; i < appendage.length; i++) {
			ret[i + data.length] = appendage[i];
		}
		return ret;
	}
	
	public static double[][] append(double[][] data, double[][] appendage) {
		if(data == null) {
			return appendage;
		}
		double[][] ret = new double[data.length + appendage.length][data[0].length];
		for(int i = 0; i < data.length; i++) {
			ret[i] = data[i];
		}
		for(int i = 0; i < appendage.length; i++) {
			ret[i + data.length] = appendage[i];
		}
		return ret;
	}
	
	public static int[] append(int[] data, int appendage) {
		if(data == null) {
			return new int[]{ appendage };
		}
	    int length = data.length;
	    int[] newarray = new int[length + 1];
	    System.arraycopy(data, 0, newarray, 0, length);
	    newarray[length] = appendage;
	    return newarray;
	}
	
	public static double[] append(double[] data, double appendage) {
		if(data == null) {
			return new double[]{ appendage };
		}
	    int length = data.length;
	    double[] newarray = new double[length + 1];
	    System.arraycopy(data, 0, newarray, 0, length);
	    newarray[length] = appendage;
	    return newarray;
	}
	
	public static String[] append(String[] data, String appendage) {
		if(data == null) {
			return new String[]{ appendage };
		}
	    int length = data.length;
	    String[] newarray = new String[length + 1];
	    System.arraycopy(data, 0, newarray, 0, length);
	    newarray[length] = appendage;
	    return newarray;
	}
	
	public static String[] append(String[] data, String[] appendage) {
		if(data == null) {
			return appendage;
		}
		String[] ret = data.clone();
		for(String a : appendage) {
			ret = append(ret, a);
		}
		return ret;
	}
	
	public static double[] append(double[] data, double[] appendage) {
		if(data == null) {
			return appendage;
		}
		double[] ret = data.clone();
		for(double a : appendage) {
			ret = append(ret, a);
		}
		return ret;
	}
	
	public static int[] prepend(int prependage, int[] data) {
		if(data == null) {
			return new int[]{ prependage };
		}
	    int length = data.length;
	    int[] newarray = new int[length + 1];
	    newarray[0] = prependage;
	    System.arraycopy(data, 0, newarray, 1, length);
	    return newarray;
	}
	
	public static Matrix append(Matrix data, Matrix appendage) {
		if(data == null) {
			return appendage;
		}
		int m1 = data.getRowDimension();
		int m2 = appendage.getRowDimension();
		int n = data.getColumnDimension();
		Matrix ret = new Matrix(m1 + m2, n);
		
		for (int i = 0; i != m1; i++) {
			for (int j = 0; j != n; j++) {
				ret.set(i, j, data.get(i, j));
			}
		}
		for (int i = 0; i != m2; i++) {
			for (int j = 0; j != n; j++) {
				ret.set(i + m1, j, appendage.get(i, j));
			}
		}
		
		//ret.setMatrix(0, m1 - 1, 0, n - 1, data);
		//ret.setMatrix(m1, m1 + m2 - 1, 0, n - 1, appendage);
		
		return ret;
	}
	
	public static double[][] copy(double[][] data) {
		if (data == null) {
			return null;
		}
		
		double[][] result = new double[data.length][data[0].length];
		for (int i = 0; i != data.length; i++) {
			System.arraycopy(data[i], 0, result[i], 0, data[0].length);
		}
		return result;
	}
	
	public static int[][] copy(int[][] data) {
		if (data == null) {
			return null;
		}
		
		int[][] result = new int[data.length][data[0].length];
		for (int i = 0; i != data.length; i++) {
			System.arraycopy(data[i], 0, result[i], 0, data[0].length);
		}
		return result;
	}
	
	// Shuffles the values of matrices vertically (must have same dimensions)
	public static Matrix intersperseColumns(Vector<Matrix> matrices) {
		int m = matrices.get(0).getRowDimension();
		int n = matrices.get(0).getColumnDimension();
		Matrix ret = new Matrix(m, n * matrices.size());
		int matNum = 0; int colNum = 0;
		for(int i = 0; i < n * matrices.size(); i++, matNum++) {
			if(matNum == matrices.size()) {
				matNum = 0;
				colNum++;
			}
			ret.setMatrix(0, m - 1 ,i, i, matrices.get(matNum).getMatrix(0, m - 1, colNum, colNum));
		}
		return ret;
	}
	
	public static Matrix appendColumn(Matrix data, Matrix appendage) {
		if(data == null) {
			return appendage;
		}
		int m = data.getRowDimension();
		int n1 = data.getColumnDimension();
		int n2 = appendage.getColumnDimension();
		Matrix ret = new Matrix(m, n1 + n2);
		
		ret.setMatrix(0, m - 1, 0, n1 - 1, data);
		ret.setMatrix(0, m - 1, n1, n1 + n2 - 1, appendage);
		
		return ret;
	}
	
	/**
	 * Returns a random permutation of integers from 0 to n
	 * @param n the upperbound
	 * @return the randomly permuted integer array
	 */
	public static int[] randomPermutations(int n) {
		int[] data = MLFuncs.range(0, n - 1);
		for(int i = 0; i < data.length; i++) {
			int swapSpace = data[i];
			int j = (int)Math.floor(Math.random() * n);
			data[i] = data[j];
			data[j] = swapSpace;
		}
		return data;
	}
	
	public static int[] lessThanOrEqualTo(int[] data, int value) {
		int[] ret = new int[data.length];
		for(int i = 0; i < data.length; i++) {
			if(data[i] <= value) {
				ret[i] = 1;
			}
		}
		return ret;
	}
	
	public static int[][] transpose(int[] data) {
		return (int[][])ArrayFuncs.convertArray(transpose((double[])ArrayFuncs.convertArray(data, double.class)), int.class);
	}
	
	public static double[][] transpose(double[] data) {
		double[][] ret = new double[data.length][1];
		for(int i = 0; i < data.length; i++) {
			ret[i][0] = data[i];
		}
		return ret;
	}
	
	public static int[][] transpose(int[][] data) {
		return (int[][])ArrayFuncs.convertArray(transpose((double[][])ArrayFuncs.convertArray(data, double.class)), int.class);
	}
	
	public static int[] flattenHorizontally(int[][] data) {
		return (int[])ArrayFuncs.convertArray(ArrayFuncs.flatten(data), int.class);
	}
	
	public static Matrix flattenHorizontally(Matrix data) {
		return new Matrix((double[])ArrayFuncs.convertArray(ArrayFuncs.flatten(data.getArray()), double.class), 1);
	}
	
	public static double[] flattenVertically(double[][][] data) {
		int m = data.length;
		int n = data[0].length;
		int o = data[0][0].length;
		
		double[] flattened = new double[m * n * o];
		int p = 0;
		for (int k = 0; k != o; k++) {
			for (int j = 0; j != n; j++) {
				for (int i = 0; i != m; i++) {
					flattened[p] = data[i][j][k];
					p++;
				}
			}
		}
		
		return flattened;
	}
	
	public static double[] flattenVertically(double[][] data) {
		int m = data.length;
		if(m == 1) {
			return data[0];
		}
		int n = data[0].length;
		double[] flattened = new double[m * n];
		int k = 0;
		for (int i = 0; i != n; i++) {
			for (int j = 0; j != m; j++) {
				flattened[k] = data[j][i];
				k++;
			}
		}
		
		return flattened;
	}
	
	public static double[] flattenVertically(Matrix data) {
		int m = data.getRowDimension();
		if(m == 1) {
			return data.getArray()[0];
		}
		int n = data.getColumnDimension();
		double[] flattened = new double[m * n];
		int k = 0;
		for (int i = 0; i != n; i++) {
			for (int j = 0; j != m; j++) {
				flattened[k] = data.get(j, i);
				k++;
			}
		}
		
		return flattened;
	}
	
	public static int[] flattenVertically(int[][] data) {
		int m = data.length;
		if(m == 1) {
			return data[0];
		}
		int n = data[0].length;
		int[] flattened = new int[m * n];
		for(int i = 0; i < n; i++) {
			System.arraycopy(getColumn(data, i), 0, flattened, m * i, m);
		}
		return flattened;
	}
	
	public static double[][] transpose(double[][] data) {
		return new Matrix(data).transpose().getArray();
	}
	
	public static int[] randInts(int size, int upperBound) {
		int[] ret = new int[size];
		for(int i = 0; i < size; i++) {
			ret[i] = (int)Math.floor(Math.random() * upperBound);
		}
		return ret;
	}
	
	public static int[] unique(int[] data) {
		HashSet<Integer> s = new HashSet<Integer>();
		for(int d : data) {
			s.add(new Integer(d));
		}
		int[] ret = new int[s.size()];
		Iterator<Integer> it = s.iterator();
		for(int i = 0; i < ret.length; i++) {
			ret[i] = it.next().intValue();
		}
		return ret;
	}
	
	public static Matrix diag(Matrix M) {
		int m = M.getRowDimension();
		int n = M.getColumnDimension();
		Matrix ret = null;
		if(m == 1) {
			ret = new Matrix(n, n);
			for(int i = 0; i < n; i++) {
				ret.set(i, i, M.get(0, i));
			}
		} else if(n == 1) {
			ret = new Matrix(m, m);
			for(int i = 0; i < m; i++) {
				ret.set(i, i, M.get(i, 0));
			}
		} else {
			ret = new Matrix(1, m);
			for(int i = 0; i < m; i++) {
				ret.set(0, i, M.get(i, i));
			}
		}
		return ret;
	}
	
	/**
	 * Returns a Matrix with ones where a >= b
	 * @param a Matrix to compare to
	 * @param b Matrix to compare
	 * @return Matrix with ones where a >= b
	 */
	public static Matrix greaterThanOrEqualTo(Matrix a, Matrix b) {
		int m = a.getRowDimension();
		int n = a.getColumnDimension();
		Matrix ret = new Matrix(m, n);
		for(int i = 0; i < m ; i++) {
			for(int j = 0; j < n; j++) {
				if(a.get(i, j) >= b.get(i, j)) {
					ret.set(i, j, 1);
				}
			}
		}
		return ret;
	}
	
	/**
	 * Given a Matrix, returns a Matrix with absolute values of each element
	 * @param M the input matrix
	 * @return |a|
	 */
	public static Matrix abs(Matrix M) {
		Matrix ret = M.copy();
		for(int i = 0; i < ret.getRowDimension(); i++) {
			for(int j = 0; j < ret.getColumnDimension(); j++) {
				ret.set(i, j, Math.abs(ret.get(i, j)));
			}
		}
		return ret;
	}
	
	public static int[] getSortedIndex(double[] data) {
		FastQuickSort f = new FastQuickSort();
		f.sort(data);
		return f.sortedIndex;
	}
	
	public static int[] getSortedIndex(int[] data) {
		FastQuickSort f = new FastQuickSort();
		f.sort((double[])ArrayFuncs.convertArray(data, double.class));
		return f.sortedIndex;
	}
	
	/** Sorts input data into ascending order using FastQuickSort.
	 * 
	 * @param data
	 * @return data in ascending order
	 */
	public static int[] sortAscending(int[] data) {
		FastQuickSort f = new FastQuickSort();
		double[] dData = (double[])ArrayFuncs.convertArray(data, double.class);
		f.sort(dData);
		data = (int[])ArrayFuncs.convertArray(dData, int.class);
		return data;
	}
	
	/** Sorts input data into ascending order using FastQuickSort.
	 * 
	 * @param data
	 * @return data in ascending order
	 */
	public static double[] sortAscending(double[] data) {
		FastQuickSort f = new FastQuickSort();
		f.sort(data);
		return data;
	}
	
	/** Sorts input data into ascending order using FastQuickSort.
	 * Returns int array containing sorted data
	 * 
	 * @param Integer array of data
	 * @return int array containing data values in ascending order
	 */
	public static int[] sortAscending(Integer[] data) {
		FastQuickSort f = new FastQuickSort();
		//double[] dData = (double[])ArrayFuncs.convertArray(data, double.class);
		double[] dData = new double[data.length];
		for (int i = 0; i < data.length; ++i) {
			dData[i] = (double)data[i];
		}
		f.sort(dData);
		int[] intData = (int[])ArrayFuncs.convertArray(dData, int.class);
		return intData;
	}
	
	public static int[] getColumn(int[][] data, int index) {
		int[] column = new int[data.length];
		
		for(int i = 0; i < data.length; i++) {
			column[i] = data[i][index];
		}
		return column;
	}

    public static double[] getColumn(double[][] data, int index) {
		double[] column = new double[data.length];
		
		for(int i = 0; i < data.length; i++) {
			column[i] = data[i][index];
		}
		return column;
	}
	
	public static double[] getColumn(Matrix data, int index) {
		int numRows = data.getRowDimension();
		double[] column = new double[numRows];
		
		for(int i = 0; i < numRows; i++) {
			column[i] = data.get(i, index);
		}
		return column;
	}
	
	// Returns the columns relative to the XY-plane for the given index.
	public static double[][] getXYColumns(double[][][] data, int index) {
		double[][] columns = new double[data.length][data[0][0].length];
		
		for(int i = 0; i < data.length; i++) {
			for (int j = 0; j < data[0][0].length; j++) {
				columns[i][j] = data[i][index][j];
			}
		}
		return columns;
	}
	
	// Returns the rows relative to the XY-plane for the given index.
	public static double[][] getXYRows(double[][][] data, int index) {
		double[][] columns = new double[data[0].length][data[0][0].length];
		
		for(int i = 0; i < data[0].length; i++) {
			for (int j = 0; j < data[0][0].length; j++) {
				columns[i][j] = data[index][i][j];
			}
		}
		return columns;
	}
	
	public static int[] getRow(int[][] data, int index) {
		return data[index];
	}
	
	public static double[] getRow(double[][] data, int index) {
		return data[index];
	}
	
	public static double[] getRow(Matrix data, int index) {
		int numColumns = data.getColumnDimension();
		double[] row = new double[numColumns];
		
		for(int i = 0; i < numColumns; i++) {
			row[i] = data.get(index, i);
		}
		return row;
	}
	
	/**
	 * Subtract val from each element of data
	 * @param data
	 * @param val
	 * @return
	 */
	public static int[] subtract(int[] data, int val) {
		if(data == null) {
			return null;
		}
		int[] ret = data.clone();
		for(int i = 0; i < ret.length; i++) {
			ret[i] -= val;
		}
		return ret;
	}
	
	public static double[] subtract(double[] a, double[] b) {
		int m = a.length;
		
		double[] ret = new double[m];
		for (int i = 0; i != m; i++) {
			ret[i] = a[i] - b[i];
		}
		
		return ret;
	}
	
	public static double[][] subtract(double[][] a, double[][] b) {
		int m = a.length;
		
		double[][] ret = new double[m][];
		for (int i = 0; i != m; i++) {
			ret[i] = subtract(a[i], b[i]);
		}
		
		return ret;
	}
	
	public static double[][] times(double[][] data, int val) {
		return new Matrix(data).times(val).getArray();
	}
	
	public static int[] times(int[] data, int val) {
		double[][] ddata = new double[][]{(double[])ArrayFuncs.convertArray(data, double.class)};
		return (int[])ArrayFuncs.convertArray(times(ddata, val)[0], int.class);
	}
	
	public static int[] product(int[] data, int val) {
		int[] ret = new int[data.length];
		for(int i = 0; i < data.length; i++) {
			ret[i] = data[i] * val;
		}
		return ret;
	}
	
	public static double[] product(double[] data, double val) {
		double[] ret = new double[data.length];
		for(int i = 0; i < data.length; i++) {
			ret[i] = data[i] * val;
		}
		return ret;
	}
	
	public static int[] sum(int[] data1, int[] data2) {
		int[] ret = new int[data1.length];
		for(int i = 0; i < data1.length; i++) {
			ret[i] = data1[i] + data2[i];
		}
		return ret;
	}
	
	public static int product(int[] data) {
		if(data == null || data.length == 0) {
			return 0;
		}
		int ret = 1;
		for(int d : data) {
			ret *= d;
		}
		return ret;
	}
	
	public static double[] divide(double[] a, double[] b) {
		int m = a.length;
		
		double[] ret = new double[m];
		for (int i = 0; i != m; i++) {
			ret[i] = a[i] / b[i];
		}
		
		return ret;
	}
	
	public static double[][] divide(double[][] a, double[][] b) {
		int m = a.length;
		
		double[][] ret = new double[m][];
		for (int i = 0; i != m; i++) {
			ret[i] = divide(a[i], b[i]);
		}
		
		return ret;
	}
	
	public static double[] divide(double[] data, double val) {
		int m = data.length;
		
		double[] ret = new double[m];
		for (int i = 0; i != m; i++) {
			ret[i] = data[i] / val;
		}
		
		return ret;
	}
	
	public static double[][] divide(double[][] data, double val) {
		int m = data.length;
		
		double[][] ret = new double[m][];
		for (int i = 0; i != m; i++) {
			ret[i] = divide(data[i], val);
		}
		
		return ret;
	}
	
	/**
	 * Remove val from the array if it exists
	 * @param data
	 * @param val
	 * @return
	 */
	public static int[] remove(int[] data, int val) {
		int[] ret = null;
		for(int d : data) {
			if(d != val) {
				ret = append(ret, d);
			}
		}
		return ret;
	}
	
	/**
	 * Remove val from the array if it exists
	 * @param data
	 * @param val
	 * @return
	 */
	public static int[] removeAll(int[] data, int[] values) {
		if(values == null) {
			return data;
		}
		int[] ret = data.clone();
		for(int v : values) {
			ret = remove(ret, v);
		}
		return ret;
	}
	
	/**
	 * Add val to each element of data
	 * @param data
	 * @param val
	 * @return
	 */
	public static int[] plus(int[] data, int val) {
		int[] ret = new int[data.length];
		for(int i = 0; i < ret.length; i++) {
			ret[i] = data[i] + val;
		}
		return ret;
	}
	
	public static int[][] minus(int[][] data, int val) {
		int[][] ret = new int[data.length][data[0].length];
		for(int i = 0; i < ret.length; i++) {
			for(int j = 0; j < ret[0].length; j++) {
				ret[i][j] = data[i][j] - val;
			}
		}
		return ret;
	}
	
	public static int[][] plus(int[][] data, int val) {
		return minus(data, -val);
	}
	
	public static Matrix square(Matrix M) {
		Matrix ret = M.copy();
		for(int i = 0; i < ret.getRowDimension(); i++) {
			for(int j = 0; j < ret.getColumnDimension(); j++) {
				double value = ret.get(i, j);
				ret.set(i, j, value * value);
			}
		}
		return ret;
	}
	
	public static Matrix columnSum(Matrix data) {
		int m = data.getRowDimension();
		int n = data.getColumnDimension();
		
		if(m == 1) {
			return data;
		}
		
		Matrix ret = new Matrix(1, n);
		for(int i = 0; i < n; i++) {
			for(int j = 0; j < m; j++) {
				ret.set(0, i, ret.get(0, i) + data.get(j, i));
			}
		}
		return ret;
	}
	
	public static double[][] columnSum(double[][] data) {
		int m = data.length;
		int n = data[0].length;
		
		if(m == 1) {
			return data;
		}
		
		double[][] ret = new double[1][n];
		for(int i = 0; i < n; i++) {
			for(int j = 0; j < m; j++) {
				ret[0][i] += data[j][i];
			}
		}
		return ret;
	}
	
	public static Matrix rowSum(Matrix data) {
		return columnSum(data.transpose()).transpose();
	}
	
	public static double sum(Matrix data) {
		return rowSum(columnSum(data)).get(0, 0);
	}
	
	public static Matrix normalizeRow(Matrix origin) {
		Matrix normalBase = sqrt(columnSum(square(origin)));
		
		int m = origin.getRowDimension();
		int n = origin.getColumnDimension();
		
		Matrix normalBase2 = new Matrix(m, n);
		
		for(int i = 0; i < m; i++) {
			normalBase2.setMatrix(i, i, 0, n - 1, normalBase);
		}
		
		return origin.arrayRightDivide(normalBase2);
	}
	
	public static Matrix normalizeColumn(Matrix origin) {
		return(normalizeRow(origin.transpose()).transpose());
	}
	
	/**
	 * Normalize Euclidean distance of vectors in original matrix to unit 1.
	 * @parm origin the original matrix
	 * @parm DIM, direction of vectors in original matrix. 1 = stands for vectors are stacked in column-wise; 2 = stands for vectors stacked row-wise.
	 * @return normal matrix
	 */
	public static Matrix normalizeEuc(Matrix brainLV, double DIM) {
		
		Matrix normal_base = null;

		if (DIM == 1){
			normal_base = sqrt(columnSum(brainLV.arrayTimes(brainLV)));
			normal_base = replicateRows(normal_base, brainLV.getRowDimension());
		}
		else if (DIM ==2) {
			normal_base = sqrt(rowSum(brainLV.arrayTimes(brainLV)));
			normal_base = replicateColumns(normal_base, brainLV.getColumnDimension());
		}
		
		//Change any zero items in normal_base to 1;
		//int [] zeroItem = find(normal_base, 0);
		//normal_base.get(zeroItem) = 1;
		
		//normal_base(zero_items) = 1;	
		return  brainLV.arrayRightDivide(normal_base);
	}
	
	
	private static void getRowDimension() {
		// TODO Auto-generated method stub
		
	}

	public static Matrix sqrt(Matrix M) {
		Matrix ret = M.copy();
		for(int i = 0; i < ret.getRowDimension(); i++) {
			for(int j = 0; j < ret.getColumnDimension(); j++) {
				double value = ret.get(i, j);
				ret.set(i, j, Math.sqrt(value));
			}
		}
		return ret;
	}
	
	/**
	 * Return indices where the value is found in the double array
	 * (0-relative)
	 * @param data the input double array
	 * @param value the value to find
	 * @return indeces where value is found in D
	 */
	public static int[] find(double[] data, double value) {
		int[] indeces = new int[data.length];
		int pos = 0;
		for(int i = 0; i < data.length; i++) {
			if(data[i] == value) {
				indeces[pos] = i;
				pos++;
			}
		}
		int[] ret = new int[pos];
		System.arraycopy(indeces, 0, ret, 0, pos);
		return ret;
	}
	
	public static int[] find(Matrix data, double value) {
		return find(flattenVertically(data), value);
	}
	
	/**
	 * Return indices where the value is greater than or equal to those found in the double array
	 * @param data the input double array
	 * @param value the value to find
	 * @return indices where value is found in D
	 */
	public static int[] findLessThanOrEqualTo(double[] data, double value) {
		int[] indeces = new int[data.length];
		int pos = 0;
		for(int i = 0; i < data.length; i++) {
			if(data[i] <= value) {
				indeces[pos] = i;
				pos++;
			}
		}
		int[] ret = new int[pos];
		System.arraycopy(indeces, 0, ret, 0, pos);
		return ret;
	}
	
	public static int[] findLessThanOrEqualTo(Matrix data, double value) {
		return findLessThanOrEqualTo(flattenVertically(data), value);
	}
	
	/**
	 * Return indices of elements greater than or equal to the input boundary value
	 * @param data the input int array
	 * @param value the boundary value 
	 * @return indices of input int array satisfying the boundary condition
	 */
	public static int[] findGreaterThanOrEqualTo(int[] data, int value) {
		int[] indices = new int[data.length];
		int pos = 0;
		for(int i = 0; i < data.length; i++) {
			if(data[i] >= value) {
				indices[pos] = i;
				pos++;
			}
		}
		int[] ret = new int[pos];
		System.arraycopy(indices, 0, ret, 0, pos);
		return ret;
	}
	
	/**
	 * Return indices of elements less than or equal to the input boundary value
	 * @param data the input int array
	 * @param value the boundary value 
	 * @return indices of input int array satisfying the boundary condition
	 */
	public static int[] findLessThanOrEqualTo(int[] data, int value) {
		int[] indices = new int[data.length];
		int pos = 0;
		for(int i = 0; i < data.length; i++) {
			if(data[i] <= value) {
				indices[pos] = i;
				pos++;
			}
		}
		int[] ret = new int[pos];
		System.arraycopy(indices, 0, ret, 0, pos);
		return ret;
	}
	
	
	/**
	 * Return indices where the value is greater than than those found in the double array
	 * @param data the input double array
	 * @param value the threshold value
	 * @return indeces of elements greater than 'value'
	 */
	public static int[] findGreaterThan(double[] data, double value) {
		int[] indeces = new int[data.length];
		int pos = 0;
		for(int i = 0; i < data.length; i++) {
			if(data[i] > value) {
				indeces[pos] = i;
				pos++;
			}
		}
		int[] ret = new int[pos];
		System.arraycopy(indeces, 0, ret, 0, pos);
		return ret;
	}
	
	/**
	 * Return indices where the value is found in the int array
	 * @param data the input double array
	 * @param value the value to find
	 * @return indices where value is found in D
	 */
	public static int[] find(int[] data, int value) {
		return (int[])ArrayFuncs.convertArray(find((double[])ArrayFuncs.convertArray(data, double.class), value), int.class);
	}
	
	/** Returns indices of non-zero elements in input array
	 * @param data 	the input double array
	 * @return int array of indices where data is non-zero
	 */
	public static int[] findNonZero(double[] data) {
		ArrayList<Integer> whereNZ = new ArrayList<Integer>();
		int dLength = data.length;
		for (int d = 0; d < dLength; ++d) {
			if (data[d] != 0) {
				whereNZ.add(d);
			}
		}
		int[] locs = new int[whereNZ.size()];
		for (int i = 0; i < locs.length; ++i) {
			locs[i] = whereNZ.get(i);
		}
		return locs;
	}
	
	/** Returns indices of non-zero elements in input array
	 * @param data 	the input int array
	 * @return int array of indices where data is non-zero
	 */
	public static int[] findNonZero(int[] data) {
		ArrayList<Integer> whereNZ = new ArrayList<Integer>();
		int dLength = data.length;
		for (int d = 0; d < dLength; ++d) {
			if (data[d] != 0) {
				whereNZ.add(d);
			}
		}
		int[] locs = new int[whereNZ.size()];
		for (int i = 0; i < locs.length; ++i) {
			locs[i] = whereNZ.get(i);
		}
		return locs;
	}
	
	/** Returns indices of non-zero elements in input array
	 * @param data 	the input byte array
	 * @return int array of indices where data is non-zero
	 */
	public static int[] findNonZero(byte[] data) {
		ArrayList<Integer> whereNZ = new ArrayList<Integer>();
		int dLength = data.length;
		for (int d = 0; d < dLength; ++d) {
			if (data[d] != 0) {
				whereNZ.add(d);
			}
		}
		int[] locs = new int[whereNZ.size()];
		for (int i = 0; i < locs.length; ++i) {
			locs[i] = whereNZ.get(i);
		}
		return locs;
	}
	
	
	public static boolean contains(int[] data, int value) {
		for(int d : data) {
			if(d == value) {
				return true;
			}
		}
		return false;
	}
	
	/**
	 * Return sum of all elements in data
	 * @param data
	 * @return
	 */
	public static double sum(double[] data) {
		double sum = 0;
		for(double i : data) {
			sum += i;
		}
		return sum;
	}
	
	/**
	 * Return sum of all elements in data
	 * @param data
	 * @return
	 */
	public static int sum(int[] data) {
		if(data == null) {
			return 0;
		}
		return (int)sum((double[])ArrayFuncs.convertArray(data, double.class));
	}
	
	/**
	 * Return minimum value in data
	 * @param data
	 * @return
	 */
	public static int min(int[] data) {
		int min = Integer.MAX_VALUE;
		for(int i : data) {
			min = Math.min(i, min);
		}
		return min;
	}
	
	/**
	 * Return minimum value in data
	 * @param data
	 * @return min value in data array
	 */
	public static double min(double[] data) {
		double min = Double.MAX_VALUE;
		for(double i : data) {
			min = Math.min(i, min);
		}
		return min;
	}
	
	/**
	 * Return maximum value in data
	 * @param data
	 * @return max value in data array
	 */
	public static double max(double[] data) {
		double max = Double.NEGATIVE_INFINITY;
		for(double i : data) {
			max = Math.max(i, max);
		}
		return max;
	}
	
	/**
	 * Return maximum value in data
	 * @param data
	 * @return max value in data array
	 */
	public static int max(int[] data) {
		int max = Integer.MIN_VALUE;
		for(int i : data) {
			max = Math.max(i, max);
		}
		return max;
	}
	
	public static double min(Matrix M) {
		int m = M.getRowDimension();
		int n = M.getColumnDimension();
		
		double min = Double.MAX_VALUE;
		for(int i = 0; i < m; i++) {
			for(int j = 0; j < n; j++) {
				min = Math.min(min, M.get(i, j));
			}
		}
		
		return min;
	}
	
	/**
	 * Return the column mean of the input Matrix
	 * @param M the input Matrix
	 * @return a row Matrix containing the 
	 *         mean of each column of M
	 */
	public static Matrix columnMean(Matrix M) {
		int m = M.getRowDimension();
		int n = M.getColumnDimension();
		
		if(m == 1) {
			return M;
		}
		
		Matrix sum = columnSum(M);
		Matrix columnMean = new Matrix(1, n);
		for(int i = 0; i < n; i++) {
			double value = sum.get(0, i);
			columnMean.set(0, i, value / m);
		}
		return columnMean;
	}
	public static double[][] columnMean(double[][] data) {
		int m = data.length;
		int n = data[0].length;
		
		if(m == 1) {
			return data;
		}
		
		double[][] sum = columnSum(data);
		double[][] columnMean = new double[1][n];
		for(int i = 0; i < n; i++) {
			double value = sum[0][i];
			columnMean[0][i] = value / m;
		}
		return columnMean;
	}
	
	/**
	 * Calculate column means, ignoring given ignoreVal.
	 * E.g. if data column is {1, 1, -1} and ignoreVal = -1
	 * then column mean = 1.
	 * If all vals in given column are to be ignored, mean = NaN.
	 * @param data
	 * @param ignoreVal
	 * @return array of column means
	 */
	public static double[] selectedColumnMean(double[][] data, double ignoreVal) {
		int nRows = data.length;
		int nCols = data[0].length;
		
		double[] colMeans = new double[nCols];
		for (int j = 0; j < nCols; ++j) {
			int count = 0;
			for (int i = 0; i < nRows; ++i) {
				if (data[i][j] != ignoreVal) {
					colMeans[j] += data[i][j];
					++count;
				}
			}
			colMeans[j] /= count;
		}
		return colMeans;
	}
	
	
	/**
	 * Calculate row means, ignoring given ignoreVal.
	 * E.g. if data row is {1, 1, -1} and ignoreVal = -1
	 * then row mean = 1.
	 * If all vals in given row are to be ignored, mean = NaN.
	 * @param data
	 * @param ignoreVal
	 * @return array of row means
	 */
	public static double[] selectedRowMean(double[][] data, double ignoreVal) {
		int nRows = data.length;
		int nCols = data[0].length;
		
		double[] rowMeans = new double[nRows];
		for (int i = 0; i < nRows; ++i) {
			int count = 0;
			for (int j = 0; j < nCols; ++j) {
				if (data[i][j] != ignoreVal) {
					rowMeans[i] += data[i][j];
					++count;
				}
			}
			rowMeans[i] /= count;
		}
		return rowMeans;
	}
	
	
	/** Returns mean of each row of input Matrix.
	 * 
	 * @param M input Matrix with m rows and n columns 
	 * @return row Matrix (1 row, m columns) containing 
	 *         mean of each row of M
	 */
	public static Matrix rowMean(Matrix M) {
		return columnMean(M.transpose());
	}
	
	public static double columnMedian(double[][] data, int column) {
		double[] values = new double[data.length];
		for (int i = 0; i != values.length; i++) {
			values[i] = data[i][column];
		}
		Arrays.sort(values);
		
		return median(values);
	}
	
	public static double rowMedian(double[][] data, int row) {
		double[] values = new double[data[0].length];
		for (int i = 0; i != values.length; i++) {
			values[i] = data[row][i];
		}
		Arrays.sort(values);
		
		return median(values);
	}
	
	public static double median(double[] data) {
		double result;
		
		data = sortAscending(data);
		// If there is an even number of values, we take the average
		// of the two middle values.
		if (data.length % 2 == 0) {
			result = data[data.length / 2] + data[(data.length / 2) - 1];
			result /= 2;
			
		// Otherwise, we take the single middle value only.
		} else {
			result = data[data.length / 2];
		}
		
		return result;
	}
	
	public static int[][] vectorToIntArray(Vector<int[][]> vec) {
		int m = 0;
		for(int[][] v: vec) {
			m += v.length;
		}
		int n = vec.get(0)[0].length;
		int[][] ret = new int[vec.size() * m][n];
		int incr = 0;
		for(int[][] v : vec) {
			for(int i = 0; i < v.length; i++) {
				ret[i + incr] = v[i];
			}
			incr += v.length;
		}
		return ret;
	}
	
	public static int[] vectorToIntArray(Vector<int[]> vec) {
		int n = 0;
		for(int[] v : vec) {
			n += v.length;
		}
		int[] ret = new int[n];
		int incr = 0;
		for(int[] v : vec) {
			for(int i = 0; i < v.length; i++) {
				ret[i + incr] = v[i];
			}
			incr += v.length;
		}
		return ret;
	}
	
	public static int[][] setColumn(int[][] data, int colNumber, int[] filler) {
		if(data == null) {
			return transpose(filler);
		}
		int[][] ret = data.clone();
		for(int i = 0; i < filler.length; i++) {
			ret[i][colNumber] = filler[i];
		}
		return ret;
	}
	
	public static double[] rangeDouble(int start, int end) {
		if(end < 0) {
			return null;
		}
		double[] ret = new double[end - start + 1];
		for(int i = 0; start < end + 1; start++, i++) {
			ret[i] = start;
		}
		return ret;
	}
	
	public static int[] range(int start, int end) {
		if(end < 0) {
			return null;
		}
		int[] ret = new int[end - start + 1];
		for(int i = 0; start < end + 1; start++, i++) {
			ret[i] = start;
		}
		return ret;
	}
	
	public static int[] range(int start, int end, int increment) {
		if(end < 0) {
			return null;
		}
		int[] ret = new int[(end - start) / increment];
		for(int i = 0; i < ret.length; start += increment, i++) {
			ret[i] = start;
		}
		return ret;
	}
	
	/**
	 * Sets the values at the given indeces from toValues to fromValues
	 * @param toValues
	 * @param indeces
	 * @param fromValues
	 * @return
	 */
	public static int[] setValues(int[] toValues, int[] indeces, int[] fromValues) {
		int[] ret = toValues.clone();
		for(int i : indeces) {
			ret[i] = fromValues[i];
		}
		return ret;
	}
	
	/** Sets the values of input array at given indices to given new values.
	 *  [Note - unlike MLFuncs.setValues(int[], int[], int[]), the array of
	 *  newVals is only as long as 'indices' array, not necessarily as long
	 *  as original array.]
	 * 
	 * @param origArray - int array whose values are to be changed
	 * @param indices - int array of locations where origArray is to be changed
	 * @param newVals - int array (same length as 'indices') of new values
	 * @return array with newVals set at input indices of origArray
	 */
	public static int[] setVals(int[] origArray, int[] indices, int[] newVals) {
		int[] ret = origArray.clone();
		for (int i = 0; i < indices.length; ++i) {
			ret[indices[i]] = newVals[i];
		}
		return ret;
	}
	
	/** Sets the values of input array at given indices to given new values.
	 *  [Note - unlike MLFuncs.setValues(int[], int[], int[]), the array of
	 *  newVals is only as long as 'indices' array, not necessarily as long
	 *  as original array.]
	 * 
	 * @param origArray - double array whose values are to be changed
	 * @param indices - int array of locations where origArray is to be changed
	 * @param newVals - double array (same length as 'indices') of new values
	 * @return array with newVals set at input indices of origArray
	 */
	public static double[] setVals(double[] origArray, int[] indices, double[] newVals) {
		double[] ret = origArray.clone();
		for (int i = 0; i < indices.length; ++i) {
			ret[indices[i]] = newVals[i];
		}
		return ret;
	}
	
	
	public static int[][] reshape(int[][] data, int x, int y)  {
		double[][] ddata = (double[][])ArrayFuncs.convertArray(data, double.class);
		double[][] dret = reshape(new Matrix(ddata), x, y).getArray();
		return (int[][])ArrayFuncs.convertArray(dret, int.class);
	}
	
	public static double[][] reshape(double[][][] data, int x, int y) {
		return reshape(flattenVertically(data), x, y);
	}
	
	public static double[][] reshape(double[][] data, int x, int y)  {
		return reshape(flattenVertically(data), x, y);
	}
	
	public static double[][] reshape(double[] data, int x, int y)  {
		double[][] ret = new double[x][y];
		int col = 0;
		for (int i = 0; i != y; i++) {
			for (int j = 0; j != x; j++) {
				ret[j][i] = data[col];
				col++;
			}
		}
		return ret;
	}
	
	public static int[][] reshape(int[] data, int x, int y)  {
		int[][] ret = new int[x][y];
		int col = 0;
		for (int i = 0; i != y; i++) {
			for (int j = 0; j != x; j++) {
				ret[j][i] = data[col];
				col++;
			}
		}
		return ret;
	}
	
	public static Matrix reshape(Matrix data, int x, int y) {
		double[] flattened = flattenVertically(data);
		
		Matrix ret = new Matrix(x, y);
		int col = 0;
		for (int i = 0; i != y; i++) {
			for (int j = 0; j != x; j++) {
				ret.set(j, i, flattened[col]);
				col++;
			}
		}
		return ret;
	}
	
	public static double[][][] reshape(double[][][] data, int x, int y, int z)  {
		return reshape(flattenVertically(data), x, y, z);
	}
	
	public static double[][][] reshape(double[][] data, int x, int y, int z)  {
		return reshape(flattenVertically(data), x, y, z);
	}
	
	/** Reshapes input 1D array into 3D array with dims [x][y][z], given
	 * size of x, y, z.  Assumes input data is in order [x + y*xdim + z*xdim*ydim].
	 * @param data - 1D array in order [x + y*xdim + z*xdim*ydim]
	 * @param x
	 * @param y
	 * @param z
	 * @return 3D array with dims [x][y][z]
	 * @see #reshapeZYX(double[], int, int, int)
	 */
	public static double[][][] reshape(double[] data, int x, int y, int z)  {
		double[][][] ret = new double[x][y][z];
		int col = 0;
		for (int i = 0; i != z; i++) {
			for (int j = 0; j != y; j++) {
				for (int k = 0; k != x; k++) {
					ret[k][j][i] = data[col];
					col++;
				}
			}
		}
		return ret;
	}
	
	/** Reshapes input 1D array into 3D array with dims [z][y][x], given
	 * size of x, y, z.  Assumes input data is in order [x + y*xdim + z*xdim*ydim].
	 * @param data - 1D array in order [x + y*xdim + *xdim*ydim]
	 * @param x - size of x dim 
	 * @param y - size of y dim
	 * @param z - size of z dim
	 * @return 3D array with dims [z][y][x]
	 * @see #reshape(double[], int, int, int)
	 */
	public static double[][][] reshapeZYX(double[] data, int x, int y, int z)  {
		double[][][] ret = new double[z][y][x];
		int col = 0;
		for (int i = 0; i != z; i++) {
			for (int j = 0; j != y; j++) {
				for (int k = 0; k != x; k++) {
					ret[i][j][k] = data[col];
					col++;
				}
			}
		}
		return ret;
	}
	
	public static double[][][] reshape(Matrix data, int x, int y, int z)  {
		return reshape(flattenVertically(data), x, y, z);
	}
	
	public static Matrix replicateRows(Matrix data, int numRows) {
		int m = data.getRowDimension();
		int n = data.getColumnDimension();
		Matrix ret = new Matrix(m * numRows, n);
		
		for(int i = 0; i < m * numRows; i += m) {
			ret.setMatrix(i, i + m - 1, 0, n - 1, data);
		}
		return ret;
	}
	
	public static int[][] replicateRows(int[] data, int numRows) {
		int[][] ret = new int[numRows][data.length];
		for(int i = 0; i < data.length; i++) {
			for(int j = 0; j < numRows; j++) {
				ret[j][i] = data[i];
			}
		}
		return ret;
	}
	
	public static Matrix replicateColumns(Matrix data, int numCols) {
		int m = data.getRowDimension();
		int n = data.getColumnDimension();
		Matrix ret = new Matrix(m, n * numCols);
		
		for(int i = 0; i < n * numCols; i += n) {
			ret.setMatrix(0, m - 1, i, i + n - 1, data);
		}
		return ret;
	}
	
	public static int[][] resize(int[] data, int x, int y) {
		return (int[][])ArrayFuncs.curl(data, new int[]{x, y});
	}
	
	public static int numberLessThanOrEqualTo(int[] data, int value) {
		int ret = 0;
		for(int d : data) {
			if(d <= value) {
				ret++;
			}
		}
		return ret;
	}
	
	public static int[] ones(int length) {
		return fillArray(length, 1);
	}
	
	public static int[][] ones(int m, int n) {
		return fillArray(m, n, 1);
	}
	
	public static int[] zeros(int length) {
		return new int[length];
	}
	
	public static int[] fillArray(int length, int value) {
		int[] ret = new int[length];
		for(int i = 0; i < length; i++) {
			ret[i] = value;
		}
		return ret;
	}
	
	public static int[][] fillArray(int m, int n, int value) {
		int[][] ret = new int[m][n];
		for(int i = 0; i < m; i++) {
			for(int j = 0; j < n; j++) {
				ret[i][j] = value;
			}
		}
		return ret;
	}
	
	public static double[] getItemsAtIndices(double[] data, int[] indices) {
		double[] ret = new double[indices.length];
		int pos = 0;
		for(int i : indices) {
			ret[pos] = data[i];
			pos++;
		}
		return ret;
	}
	
	public static int[] getItemsAtIndices(int[] data, int[] indices) {
		if(indices == null) { 
			return null;
		}
		int[] ret = new int[Math.min(data.length, indices.length)];
		int pos = 0;
		for(int i = 0; i < indices.length && i < data.length; i++, pos++) {
			ret[pos] = data[indices[i]];
		}
		return ret;
	}
	
	public static Matrix getColumns(Matrix data, int[] indeces) {
		int m = data.getRowDimension();
		return data.getMatrix(0, m - 1, indeces);
	}
	
	public static Matrix getRows(Matrix data, int[] indeces) {
		int n = data.getColumnDimension();
		return data.getMatrix(indeces, 0, n - 1);
	}
	
	public static int[][] getRows(int[][] data, int[] indeces) {
		int[][] ret = new int[indeces.length][data[0].length];
		for(int i = 0; i < indeces.length; i++) {
			ret[i] = data[indeces[i]];
		}
		return ret;
	}
	
	public static double[][] getRows(double[][] data, int[] indeces) {
		double[][] ret = new double[indeces.length][data[0].length];
		for(int i = 0; i < indeces.length; i++) {
			ret[i] = data[indeces[i]];
		}
		return ret;
	}
	
	// This method converts a (1 x n) cell to a String array.
	public static String[] MLCell1dRow2StrArray(MLCell data) {
		ArrayList<String> temp = new ArrayList<String>();
		for(int c = 0; c < data.getN(); c++) {
			String thisc = ((MLChar)data.get(c)).contentToString();
			int first = thisc.indexOf('\'') + 1;
			int last = thisc.lastIndexOf('\'');
			temp.add(thisc.substring(first, last));
		}
		
		String[] ret = new String[temp.size()];
		temp.toArray(ret);
		
		return ret;
	}
	
	// This method converts an (1 x n) cell to an ArrayList of Strings.
	public static ArrayList<String> MLCell1dRow2StrArrayList(MLCell data) {
		ArrayList<String> ret = new ArrayList<String>();
		
		for (int c = 0; c < data.getN(); ++c) {
			String thisc = ((MLChar)data.get(c)).contentToString();
			int first = thisc.indexOf('\'') + 1;
			int last = thisc.lastIndexOf('\'');
			ret.add(thisc.substring(first, last));
		}
		
		return ret;
	}
	
	// This method converts an (m x 1) cell to a String array.
	public static String[] MLCell1dCol2StrArray(MLCell data) {
		ArrayList<String> temp = MLFuncs.MLCell1dRow2StrArrayList(data);
		
		String[] ret = new String[temp.size()];
		temp.toArray(ret);
		
		return ret;
	}
	
	public static int[][] getColumns(int[][] data, int[] indeces) {
		int[][] ret = new int[data.length][indeces.length];
		int count = 0;
		for(int i : indeces) {
			for(int j = 0; j < data.length; j++) {
				ret[j][count] = data[j][i];
			}
			count++;
		}
		return ret;
	}
	
	public static double[][] getColumns(double[][] data, int[] indices) {
		double[][] ret = new double[data.length][indices.length];
		int count = 0;
		for(int i : indices) {
			for(int j = 0; j < data.length; j++) {
				ret[j][count] = data[j][i];
			}
			count++;
		}
		return ret;
	}
	
	public static String[] getItemsAtIndices(String[] data, int[] indices) {
		String[] ret = new String[indices.length];
		int pos = 0;
		for(int i : indices) {
			ret[pos] = data[i];
			pos++;
		}
		return ret;
	}
	
	// Returns the XY-planes of the given 3-dimensional array 
	// corresponding to the given z indices.
	public static double[][][] getXYArrays(double[][][] data, int[] zIndices) {
		double[][][] ret = new double[data.length][data[0].length][zIndices.length];
		int count = 0;
		for (int k : zIndices) {
			for (int i = 0; i < data.length; i++) {
				for (int j = 0; j < data[0].length; j++) {
					ret[i][j][count] = data[i][j][k];
				}
			}
			count++;
		}
		return ret;
	}
	
	/** Input data array is sorted in place and returned.  Rows are 
	 * sorted in order or input int array of indices into orig array.
	 * @param data
	 * @param sortedIndices
	 * @return
	 */
	public static Double[][] sortRows(Double[][] data, int[] sortedIndices) {
		int nRows = data.length;
		Double[][] tmpSortedData = new Double[nRows][data[0].length];
		for (int i = 0; i < nRows; ++i) {
			tmpSortedData[i] = data[sortedIndices[i]];
		}
		data = tmpSortedData;
		return data;
	}
	
	/** Returns a new Integer array consisting of the values from 
	 * the given start value to the given end value (inclusive).
	 * @param the starting value in the array to be returned.
	 * @param the ending value in the array to be returned.
	 */
	public static Integer[] createIntegerArray(int start, int end) {
		Integer[] result = new Integer[end - start + 1];
		for (int i = 0; i != result.length; i++) {
			result[i] = new Integer(i + start);
		}
		return result;
	}
	
	/** Returns mean of input vals
	 * @param double[] vals
	 * @return double - mean of input vals
	 */
	public static double avg(double[] vals) {
		int size = vals.length;
		double mean = 0.0;
		for (double v : vals) {
			mean += v/size;
		}
		return mean;
	}
	
	/** Returns mean of input vals
	 * @param float[] vals
	 * @return float - mean of input vals
	 */
	public static float avg(float[] vals) {
		int size = vals.length;
		float mean = 0;
		for (float v : vals) {
			mean += v/size;
		}
		return mean;
	}
	
	public static double stdev(double[] vals, boolean biased) {
		double mean = avg(vals);
		return stdev(mean, vals, biased);
	}
	
	/** Returns standard deviation of input vals
	 * @param double means - the avg value of vals
	 * @param double[] vals - must be at least of length 2
	 * @param boolean biased - true if dividing by N and false
	 * if dividing by N - 1, where N is the length of vals
	 * @return double - standard deviation of input vals
	 */
	public static double stdev(double mean, double[] vals, boolean biased) {
		int size = vals.length;
		
		double stdev = 0;
		for (double v : vals) {
			stdev += ((v - mean) * (v - mean));
		}
		
		if (biased) {
			stdev /= size;
		} else {
			stdev /= (size - 1);
		}
		stdev = Math.sqrt(stdev);
		
		return stdev;
	}
	
	public static double stdev(float[] vals, boolean biased) {
		float mean = avg(vals);
		return stdev(mean, vals, biased);
	}
	
	/** Returns standard deviation of input vals
	 * @param float mean - the avg value of vals
	 * @param float[] vals - must be at least of length 2
	 * @param boolean biased - true if dividing by N and false
	 * if dividing by N - 1, where N is the length of vals
	 * @return double - standard deviation of input vals
	 */
	public static double stdev(float mean, float[] vals, boolean biased) {
		int size = vals.length;
		
		double stdev = 0;
		for (float v : vals) {
			stdev += ((v - mean) * (v - mean));
		}
		
		if (biased) {
			stdev /= size;
		} else {
			stdev /= (size - 1);
		}
		stdev = Math.sqrt(stdev);
		
		return stdev;
	}
	
	/**
	 * Calculates the dot product of the two given vectors,
	 * in the form of int arrays. d1 and d2 must have the
	 * same length.
	 * @param d1 the int array representing the first vector
	 * @param d2 the int array representing the second vector
	 * @return the dot product of the two given vectors,
	 * in the form of int arrays.
	 */
	public static int dot(int[] d1, int[] d2) {
		int result = 0;
		for (int i = 0; i != d1.length; i++) {
			result += d1[i] * d2[i];
		}
		return result;
	}
	
	/**
	 * Calculates the dot product of the two given vectors,
	 * in the form of double arrays. d1 and d2 must have the
	 * same length.
	 * @param d1 the double array representing the first vector
	 * @param d2 the double array representing the second vector
	 * @return the dot product of the two given vectors,
	 * in the form of double arrays.
	 */
	public static double dot(double[] d1, double[] d2) {
		double result = 0;
		for (int i = 0; i != d1.length; i++) {
			result += d1[i] * d2[i];
		}
		return result;
	}
	
	/**
	 * Calculates the cross-correlation of the two given 2D arrays. The
	 * code here is a direct translation of the Matlab version of the
	 * function.
	 */
	public static double[][] rri_xcor(double[][] design, double[][] datamat) {
		double[][] newDesign = copy(design);
		double[][] newDatamat = copy(datamat);
		
		int r = newDatamat.length;
		double[] avg = columnMean(newDatamat)[0];
		double[] stdev = getRow(std(new Matrix(newDatamat)), 0);
		int[] checknan = find(stdev, 0);
		
		if (checknan.length != 0) {
			for (int i = 0; i != r; i++) {
				for (int j = 0; j != checknan.length; j++) {
					newDatamat[i][checknan[j]] = 0;
				}
			}
			
			for (int i = 0; i != checknan.length; i++) {
				avg[checknan[i]] = 0;
				stdev[checknan[i]] = 1;
			}
		}
		
		for (int i = 0; i != r; i++) {
			newDatamat[i] = subtract(newDatamat[i], avg);
			newDatamat[i] = divide(newDatamat[i], stdev);
		}
		
		r = newDesign.length;
		avg = columnMean(newDesign)[0];
		stdev = getRow(std(new Matrix(newDesign)), 0);
		checknan = find(stdev, 0);
		
		if (checknan.length != 0) {
			for (int i = 0; i != r; i++) {
				for (int j = 0; j != checknan.length; j++) {
					newDesign[i][checknan[j]] = 0;
				}
			}
			
			for (int i = 0; i != checknan.length; i++) {
				avg[checknan[i]] = 0;
				stdev[checknan[i]] = 1;
			}
		}
		
		for (int i = 0; i != r; i++) {
			newDesign[i] = subtract(newDesign[i], avg);
			newDesign[i] = divide(newDesign[i], stdev);
		}
		
		Matrix designMatrix = new Matrix(newDesign).transpose();
		Matrix datamatMatrix = new Matrix(newDatamat);
		Matrix xprod = designMatrix.times(datamatMatrix);
		
		return divide(xprod.getArray(), r - 1);
	}
	
	/**
	 * Creates image-wide correlation map for k scans with the data in behav.
	 * The code here is a direct translation of the Matlab version of the
	 * function.
	 */
	public static double[][] rri_corr_maps(double[][] behav, double[][] datamat, int n, int k) {
		double[][] maps = null;

		for (int i = 1; i <= k; i++) {
			int start = 1 + (n * (i - 1)) - 1;
			int end = n * i - 1;
			double[][] newBehav = getRows(behav, range(start, end));
			double[][] newDatamat = getRows(datamat, range(start, end));
			double[][] temp = rri_xcor(newBehav, newDatamat);

			maps = append(maps, temp);
		}
		
		return maps;
	}
	
	public static ArrayList<Integer> intersectionSortedArrays(int[] a, int[] b) {
		ArrayList<Integer> result = new ArrayList<Integer>();
		int i = 0;
		int j = 0;
		while (i < a.length && j < b.length) {
			if (a[i] < b[j]) {
				i++;
			} else if (a[i] > b[j]) {
				j++;
			} else {
				result.add(a[i]);
				i++;
				j++;
			}
		}
		return result;
	}
	
	public static ArrayList<Integer> unionSortedArrays(int[] a, int[] b) {
		ArrayList<Integer> result = new ArrayList<Integer>();
		int i = 0;
		int j = 0;
		while (i < a.length && j < b.length) {
			if (a[i] < b[j]) {
				result.add(a[i]);
				i++;
			} else if (a[i] > b[j]) {
				result.add(b[j]);
				j++;
			} else {
				result.add(a[i]);
				i++;
				j++;
			}
		}
		if (i == a.length) {
			while (j < b.length) {
				result.add(b[j]);
				j++;
			}
		}
		if (j == b.length) {
			while (i < a.length) {
				result.add(a[i]);
				i++;
			}
		}
		return result;
	}

	/** Updates all values in the given double array such that each
	 *  one is between 0 and 1 (inclusive).
	 */
	public static double[] normalize(double[] data) {
		double[] result = new double[data.length];
				
		double max = Double.MIN_VALUE;
		double min = Double.MAX_VALUE;
		for (double d : data) {
			if (d > max) {
				max = d;
			}
			if (d < min) {
				min = d;
			}
		}
		
		double diff = max - min;

		for (int i = 0; i != data.length; i++) {
			result[i] = (data[i] - min) / diff;
		}
		
		return result;
	}
	
	public static void main(String[] args) {
		double[] vals1 = {1.2, -2.3, 4.66, -9, -0.0004, 23.02, 99, 34.5};
		double[] vals2 = {1.3, -1.44, 5.9, -8.6, 1.2, 29.5, 100, 35.66};
		double corrCoeff = corr(vals1, vals2);
		NpairsjIO.print(vals1);
		NpairsjIO.print(vals2);
		System.out.println("Corr coeff: " + corrCoeff);;
		
	}

	public static int min(Integer[] data) {
		int min = Integer.MAX_VALUE;
		for(int i : data) {
			min = Math.min(i, min);
		}
		return min;
	}

}