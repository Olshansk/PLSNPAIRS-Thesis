package npairs.utils;

import java.io.IOException;
//import java.util.Enumeration;
import java.util.Hashtable;
//import java.util.Vector;
//
//import org.apache.commons.math.MathException;
//import org.apache.commons.math.distribution.ChiSquaredDistributionImpl;

import npairs.NpairsjException;
import npairs.io.NpairsjIO;
//import npairs.shared.matlib.EigenvalueDecomposition;
//import npairs.shared.matlib.MatlabMatrix;
import npairs.shared.matlib.Matrix;
import npairs.shared.matlib.MatrixImpl;
import pls.shared.MLFuncs;

public class JavaQuadCVA extends CVA {
//	private double rmsError;
//	private Matrix cvaEvalMat;
//	private double[] cvaEvals;
//	private Matrix cvaEvects;
//	private Matrix cvaScores;   // contains projection of input data onto cv
								// eigenimages,
								// i.e., representation of input data in cv space;
								// num rows = num rows in input data;
								// num cols = num CV dims
//	private Matrix cvaEigims = null;
//	private Matrix chiSqrInfo;
	private double[][] confidReg;
//	private int nSigDim;
//	private int[] classLabels;
//	private int nCVDim;
//	private int nClasses;
//	private int[] clsSz; // contains class sizes for each class label (class labels
						 // in ascending order)
//	private int nVol;
	private boolean debug = false;
	
	// Grigori: covariance matrices of the two classes
	private Matrix inv_S0;
	private Matrix inv_S1;
	//private double logdet_S1;
	//private double logdet_S0;
	
	
	// Grigori: mean vectors of the two classes
	private Matrix m0;
	private Matrix m1;	
	private Matrix m; // total mean
	private double CV_offset; // constant offset in calculation of CV scores
	
	// Grigori: PCA basis vectors
	private Matrix basis;

	/**
	 * Constructor for CVA object. Calculates CVA of given input. Input Matrix
	 * is modified (column mean centred).
	 * 
	 * @param data -
	 *            Matrix with rows == observations, cols = variables
	 * @param labels -
	 *            int array (length no. rows in data Matrix) containing class
	 *            label for corresp. row of data Matrix. Each class group must
	 *            contain at least 2 samples (rows).
	 * 
	 * 
	 */
	public JavaQuadCVA(Matrix data, int[] labels) throws NpairsjException {
		System.out.println("Quad CVA!");
		if (data.numRows() != labels.length) {
			throw new IllegalArgumentException(
					"No. rows in input Matrix does not match "
							+ "length of input array");
		}

		System.out.println("Data matrix: " + data.numRows() + " by " + data.numCols());
		
		final double CHISQR_THRESH = 0.95;
		classLabels = labels;
		nVol = classLabels.length;
		nCVDim = 1;
		
		double sTime = System.currentTimeMillis();
		//computeCVA(data);
		compute_covariances (data);
		compute_scores (data);
		if (debug) {
			double tTime = (System.currentTimeMillis() - sTime) / 1000;
			System.out.println("Total time CVA: " + tTime + " s");
		}
		
		// we have 2 classes and 1 CVA dimension
		nClasses = 2;
		nCVDim = 1;
		// set confidReg to zeros
		confidReg = new double[nClasses][nCVDim];
		confidReg[0][0] = 0;
		confidReg[1][0] = 0;
		// and chiSqrInfo as well
		double[][] three_zeros = new double[3][nCVDim + 1];
		three_zeros[0][0] = 0;
		three_zeros[1][0] = 0;
		three_zeros[2][0] = 0;
		chiSqrInfo = new MatrixImpl(three_zeros).getMatrix();	
		
		// DEBUGGING
		if (data.numCols() == 10) {
			data.printToFile("training", "IDL");
			try {
				NpairsjIO.printRowToIDLFile(classLabels, "classlabels");
			} catch (IOException e) {			
			}
		}
	}

	public JavaQuadCVA() {};
	
	// Grigori: new methods
	private void compute_covariances (Matrix data)  {
		Matrix S0;
		Matrix S1;
		Matrix data_vector;
		Hashtable<Integer, int[]> clsIndices = getLabelIndices(classLabels);		
        //		 TODO: Using pls MLFuncs code here but not currently in
		//       getLabelIndices for same function 
		int[] sortedUniqClasses = MLFuncs.sortAscending(MLFuncs
				.unique(classLabels));
	    int nCols = data.numCols();		
	    
//	    // mean-center the data
//	    m = (new MatrixImpl(1, nCols).getMatrix());
//	    m.setRow (0, data.colMeans());
//	    data_vector = (new MatrixImpl(1, data.numCols()).getMatrix());
//	    for (int i = 0; i < data.numRows(); i ++) {
//	    	data_vector.setRow(0, data.getRow(i));
//	    	data_vector.minusEquals(m);
//	    	data.setRow(i, data_vector.getRow(0));
//	    }

		S0 = (new MatrixImpl(nCols, nCols).getMatrix());
		int[] class0_idx = clsIndices.get(sortedUniqClasses[0]);
		Matrix currClsData = data.subMatrixRows(class0_idx);
		m0 = (new MatrixImpl(1, nCols).getMatrix());
		m0.setRow (0, currClsData.colMeans());
		m0 = m0.transpose();
		currClsData = currClsData.meanCentreColumns();
		S0 = currClsData.sspByCol();
		inv_S0 = S0.inverse();
		//logdet_S0 = Math.log (S0.det());
		System.out.println("Data matrix: " + data.numRows() + " by " + data.numCols());

		// DEBUGGING
		//if (data.numCols() == 10)
		//	currClsData.printToFile("training0", "IDL");		
		
		
		S1 = (new MatrixImpl(nCols, nCols).getMatrix());
		int[] class1_idx = clsIndices.get(sortedUniqClasses[1]);
		currClsData = data.subMatrixRows(class1_idx);
		m1 = (new MatrixImpl(1, nCols).getMatrix());
		m1.setRow (0, currClsData.colMeans());
		m1 = m1.transpose();		
		currClsData = currClsData.meanCentreColumns();
		S1 = currClsData.sspByCol();
		inv_S1 = S1.inverse();
		//logdet_S1 = Math.log (S1.det());
		
		// DEBUGGING
		//if (data.numCols() == 10)
		//	currClsData.printToFile("training1", "IDL");
		
		
		Matrix temp1 = m1.transpose().mult(inv_S1.mult(m1));
		Matrix temp0 = m0.transpose().mult(inv_S0.mult(m0));
		CV_offset = temp0.get(0,0) - temp1.get(0,0);
		
		// calculate squared norm of the mean difference vector; it is used in this.cvaEvals
		Matrix mean_diff = m1.minus(m0); 
		double[] mean_diff_values = mean_diff.getColumn(0);
		double sq_norm = 0;
		for (int i = 0; i < mean_diff_values.length; i ++) {
			sq_norm += mean_diff_values[i] * mean_diff_values[i];
		}
		cvaEvals = new double [nCols];
		cvaEvals [0] = sq_norm;
		for (int i = 1; i < nCols; i ++) {
			cvaEvals [i] = 0;
		}
		cvaEvectsSmall = mean_diff.transpose();
		rmsError = 0;
	}
	
	private void compute_scores (Matrix data) {
		Matrix dist1;
		Matrix dist0;
		Matrix temp1;
		Matrix temp0;
		Matrix data_vector;
		Matrix scores_vector;
		int i;
		
		scores_vector = new MatrixImpl(data.numRows(), 1).getMatrix();
		data_vector = (new MatrixImpl(data.numCols(), 1).getMatrix());
		for (i = 0; i < data.numRows(); i++) {				
			data_vector.setColumn(0, data.getRow(i));
			//data_vector = data_vector.transpose();

//			System.out.println("transposed data vector: " + data_vector.numRows() + " by " + data_vector.numCols());
//			System.out.println("mean: " + m1.numRows() + " by " + m1.numCols());
			
			dist1 = data_vector.minus(m1);
			dist0 = data_vector.minus(m0);
			temp1 = dist1.transpose().mult(inv_S1).mult(dist1);
			temp0 = dist0.transpose().mult(inv_S0).mult(dist0);
			temp0.minusEquals(temp1);
			
			// NOTE: don't add the difference of determinants (this introduces subject effects)
			//scores_vector.set (i, 0, temp0.get(0,0) + logdet_S0 - logdet_S1);
			//scores_vector.set (i, 0, temp0.get(0,0)); 
			// HACK: multiply the scores by 10
			scores_vector.set (i, 0, temp0.get(0,0) * 10); // + logdet_S0 - logdet_S1);
		}
		cvaScores = scores_vector;
	}
	
	private void temp_compute_scores (Matrix data) {
		Matrix temp0;
		Matrix temp1;
		Matrix temp2;
		Matrix temp3;
		Matrix data_vector;
		int i;
		int nRows = data.numRows();
		
		temp0 = inv_S0.mult(m0);
		temp1 = inv_S1.mult(m1);
		temp0 = temp1.minus(temp0);
		temp1 = data.mult(temp0);
		temp1 = temp1.mult(2.0);
		
		temp3 = new MatrixImpl(nRows, 1).getMatrix();
		temp0 = inv_S1.minus(inv_S0);
		data_vector = (new MatrixImpl(1, data.numCols()).getMatrix());
		for (i = 0; i < nRows; i++) {				
			data_vector.setRow(0, data.getRow(i));
			temp2 = data_vector.mult(temp0);
			temp2 = temp2.mult(data_vector.transpose());
			temp3.set (i, 0, temp2.get(0,0));
		}

		cvaScores = temp1.minus(temp3);
		cvaScores.plusEquals(CV_offset);
	}
	
	/** Overrides CVA.createFeatSpEigenimages(...)
	 */
	public void createFeatSpEigenimages(Matrix basisVects, boolean basisVectsInOrigSpace) {
		Matrix temp0;
		Matrix temp1;
		Matrix temp2;
		Matrix temp3;
		Matrix basis_vector;
		int i;
		int nRows = basisVects.numRows();
		
		this.basis = basisVects;
		
		if (basisVects != null) {
			temp0 = inv_S0.mult(m0);
			temp1 = inv_S1.mult(m1);
			temp0 = temp1.minus(temp0);
			temp1 = basisVects.mult(temp0);
			temp1 = temp1.mult(2.0);
			
			temp3 = new MatrixImpl(nRows, 1).getMatrix();
			temp0 = inv_S1.minus(inv_S0);
			basis_vector = (new MatrixImpl(1, basisVects.numCols()).getMatrix());
			for (i = 0; i < nRows; i++) {				
				basis_vector.setRow(0, basisVects.getRow(i));
				temp2 = basis_vector.mult(temp0);
				temp2 = temp2.mult(basis_vector.transpose());
				temp3.set (i, 0, Math.abs(temp2.get(0,0)));
				
			}

			cvaFeatSpEigims = temp1.minus(temp3);
			cvaInOrigSpace = basisVectsInOrigSpace;
//			cvaEigims = basisVects.mult(cvaEvectsSmall);
			
		} else {
			cvaFeatSpEigims = cvaEvectsSmall;
		}
	}

	
	/** Overrides CVA.calcTestCVScores(...)
	 * Calculates scores of the test data. Added by Grigori on Feb 24, 2009.
	 * 
	 *  @param TestData -
	 *  	Matrix of data points
	 */
	
	public Matrix calcTestCVScores (Matrix testData) {
		Matrix dist1;
		Matrix dist0;
		Matrix temp1;
		Matrix temp0;
		Matrix data_vector;
		Matrix scores_vector;
		Matrix PC_vector; // data vector in PC space
		int i;
		
		// DEBUGGING
		//testData.mult(basis).printToFile("test", "IDL");
		
		
		scores_vector = new MatrixImpl(testData.numRows(), 1).getMatrix();
		data_vector = (new MatrixImpl(1, testData.numCols()).getMatrix());
		for (i = 0; i < testData.numRows(); i++) {
			
			data_vector.setRow(0, testData.getRow(i));
			//PC_vector = data_vector.mult(basis).minus(m).transpose();
			PC_vector = data_vector.mult(basis).transpose();

//			System.out.println("transposed data vector: " + data_vector.numRows() + " by " + data_vector.numCols());
//			System.out.println("mean: " + m1.numRows() + " by " + m1.numCols());
			
			dist1 = PC_vector.minus(m1);
			dist0 = PC_vector.minus(m0);
			temp1 = dist1.transpose().mult(inv_S1).mult(dist1);
			temp0 = dist0.transpose().mult(inv_S0).mult(dist0);
			temp0.minusEquals(temp1);
			// NOTE: don't add the difference of determinants (this introduces subject effects)
			//scores_vector.set (i, 0, temp0.get(0,0) + logdet_S0 - logdet_S1);
			//scores_vector.set (i, 0, temp0.get(0,0)); 
			// HACK: multiply the scores by 10
			scores_vector.set (i, 0, temp0.get(0,0) * 10); // + logdet_S0 - logdet_S1);
		}		
		return scores_vector;
		
	}
	

//	// TODO: Move getLabelIndices(int[]) out of CVA, since also used in
//	// ResultSaver,
//	// Resampler and Analysis.
//	/**
//	 * Returns Hashtable containing index info corresp. to input group labels.
//	 * 
//	 * @param labels -
//	 *            int array of group labels
//	 * @return Hashtable<Integer, int[]> where key == group label and value ==
//	 *         array of ints containing indices of input array 'labels' corresp.
//	 *         to given group label.
//	 */
//	public static Hashtable<Integer, int[]> getLabelIndices(int[] labels) {
//		Hashtable<Integer, Integer> grpSizes = new Hashtable<Integer, Integer>();
//		Hashtable<Integer, Vector<Integer>> grpIndices = new Hashtable<Integer, Vector<Integer>>();
//		Hashtable<Integer, int[]> grpIndexArrays = new Hashtable<Integer, int[]>();
//
//		// get size and indices of each unique group
//		for (int row = 0; row < labels.length; ++row) {
//			try {
//				grpSizes.put(labels[row], grpSizes.get(labels[row]) + 1);
//				grpIndices.get(labels[row]).add(row);
//			} catch (NullPointerException e) {
//				grpSizes.put(labels[row], 1);
//				grpIndices.put(labels[row], new Vector<Integer>());
//				grpIndices.get(labels[row]).add(row);
//			}
//		}
//
//		// stuff grp index info into int[] arrays instead of Vectors
//		for (Enumeration uniqLabels = grpIndices.keys(); uniqLabels
//				.hasMoreElements();) {
//			Integer currLabel = (Integer) uniqLabels.nextElement();
//			int currGrpSize = grpSizes.get(currLabel).intValue();
//			Vector<Integer> currGrpIndexVec = (Vector<Integer>) (grpIndices
//					.get(currLabel));
//			Integer[] tmpCurrGrpIndices = currGrpIndexVec
//					.toArray(new Integer[currGrpSize]);
//			int[] currGrpIndices = new int[currGrpSize];
//			for (int i = 0; i < currGrpSize; ++i) {
//				currGrpIndices[i] = tmpCurrGrpIndices[i].intValue();
//			}
//			grpIndexArrays.put(currLabel, currGrpIndices);
//		}
//
//		return grpIndexArrays;
//	}


//	public double[] getEvals() {
//		return cvaEvals;
//	}	
//	
//	public Matrix getCVScores() {
//		return cvaScores;
//	}	
//	
//	public Matrix getEigims() {
//		return cvaEigims;
//	}
	
//	// Set method(s):
//	public void setEigims(Matrix newEigims) {
//		if ((newEigims.numCols() != cvaEigims.numCols())) {
//			throw new IllegalArgumentException(
//					"Input Matrix must have same number "
//							+ "of columns (no. CV Dims) as CVA eigenimages");
//		}
//		cvaEigims = newEigims;
//	}
//	

	// Why not test and possibly negate eigims/scores for each split?  
	// And why go through cvaEvals 'permutation' below when there is only ever
	// one cv dim in quad cva analysis?
	public void negateAndPermuteDims(int[] sign, int[] permutIndex)
	throws NpairsjException {
		// basically, do nothing.
		//System.out.println("Entering neg_and_permute: length(sign) = " + sign.length + ", sign(0) = " + sign[0] + "length(permutIndex) = " + permutIndex.length + "permutIndex(0) = " + permutIndex[0]);
		
		double[] tmpEvals = new double[nCVDim];
		for (int dim = 0; dim < nCVDim; ++dim) {
			tmpEvals[dim] = cvaEvals[permutIndex[dim]];
		}
		cvaEvals = tmpEvals;
		//cvaEvalMat.diag(cvaEvals);
	}
		
	
	/** Wrapper for JavaQuadCVA.saveCVAResultsIDL(String, boolean, int, int).  
	 *  Note saveR2 is not currently used in JavaQuadCVA.saveCVAResultsIDL
	 *  because no R2 is calculated.
	 * @see saveCVAResultsIDL(String, boolean, int, int)
	 */
	public void saveCVAResultsIDL(String cvaSavePref, boolean saveAsSplit,
			int splitNum, int splitHalf, boolean saveR2) throws IOException {
		saveCVAResultsIDL(cvaSavePref, saveAsSplit, splitNum, splitHalf);
	}
	
	
	/**
	 * Saves CVA results to file. If original data has been transformed into new
	 * vector space (e.g. via pca), then this method does NOT rotate data back
	 * into the original space first (must do outside of this method)
	 * 
	 * @param cvaSavePref
	 *            prefix (including path) of saved CVA files
	 * @param saveAsSplit
	 * @param splitNum
	 * @param splitHalf
	 */
	// TODO: only save input number of CV dims.
	public void saveCVAResultsIDL(String cvaSavePref, boolean saveAsSplit,
			int splitNum, int splitHalf) throws IOException {
		
		
		String cvaEvalFile;
		String saveFormat = "IDL";
		String cvaEigimFile;
		String cvaScoreFile;
		String cvaRMSFile;
		String cvaClassInfoFile;
		String cvaChiSqrInfoFile;
		String cvaConfidRegFile;
		String cvaEvectFile;

		if (saveAsSplit) {
			cvaEvalFile = cvaSavePref + ".CVA." + splitNum + "." + splitHalf
					+ ".eigval";
			cvaEigimFile = cvaSavePref + ".CVA." + splitNum + "." + splitHalf
					+ ".eigim";
			cvaScoreFile = cvaSavePref + ".CVA." + splitNum + "." + splitHalf
					+ ".can";
			cvaRMSFile = cvaSavePref + ".CVA." + splitNum + "." + splitHalf
					+ ".rms";
			cvaClassInfoFile = cvaSavePref + ".CVA." + splitNum + "."
					+ splitHalf + ".group";
			cvaChiSqrInfoFile = cvaSavePref + ".CVA." + splitNum + "."
					+ splitHalf + ".chi";
			cvaEvectFile = cvaSavePref + ".CVA." + splitNum + "." + splitHalf
					+ ".evect";
			cvaConfidRegFile = cvaSavePref + ".CVA." + splitNum + "."
					+ splitHalf + ".confidReg";
		} else {
			cvaEvalFile = cvaSavePref + ".CVA.ALL.eigval";
			cvaEigimFile = cvaSavePref + ".CVA.ALL.eigim";
			cvaScoreFile = cvaSavePref + ".CVA.ALL.can";
			cvaRMSFile = cvaSavePref + ".CVA.ALL.rms";
			cvaClassInfoFile = cvaSavePref + ".CVA.ALL.group";
			cvaChiSqrInfoFile = cvaSavePref + ".CVA.ALL.chi";
			cvaConfidRegFile = cvaSavePref + ".CVA.ALL.confidReg";
			cvaEvectFile = cvaSavePref + ".CVA.ALL.evect";
		}

		NpairsjIO.printToIDLFile(cvaEvals, cvaEvalFile);

		cvaEigimsBig.printToFile(cvaEigimFile, saveFormat);
		cvaScores.printToFile(cvaScoreFile, saveFormat);
		double[] rmsErrArray = { rmsError };
		NpairsjIO.printToIDLFile(rmsErrArray, cvaRMSFile);
		NpairsjIO.printRowToIDLFile(classLabels, cvaClassInfoFile);
		chiSqrInfo.printToFile(cvaChiSqrInfoFile, "IDL");
		// (note: confidreg info not normal idl npairs output)
		MatrixImpl mImpl = new MatrixImpl(confidReg);
		mImpl.getMatrix().printToFile(cvaConfidRegFile, "IDL");
		cvaEvectsSmall.printToFile(cvaEvectFile, saveFormat);
	}

//	public int getNumCVDims() {
//		return nCVDim;
//	}
	
//	public Matrix avgCVScores() {
//		Hashtable<Integer, int[]> condIndices = getLabelIndices(classLabels);
//		int numCond = condIndices.size();
//		int numCVDims = cvaScores.numCols();
//		int[] sortedUniqCondLabels = new int[numCond];
//		int i = 0;
//		for (Enumeration uniqCondLabels = condIndices.keys(); uniqCondLabels.hasMoreElements(); ) {
//			sortedUniqCondLabels[i] = (Integer)uniqCondLabels.nextElement();
//			++i;
//		}
//		sortedUniqCondLabels = MLFuncs.sortAscending(sortedUniqCondLabels);
//		
//		MatrixImpl avgScoresImpl = new MatrixImpl(numCond, numCVDims);
//		Matrix avgScores = avgScoresImpl.getMatrix();
//		for (int cond = 0; cond < numCond; ++cond) {
//			int[] currCondIndices = condIndices.get(sortedUniqCondLabels[cond]);
//			for (int dim = 0; dim < numCVDims; ++dim) {
//				double[] currScores = MLFuncs.getItemsAtIndices(cvaScores.getColumn(dim),
//						currCondIndices);
//				avgScores.set(cond, dim, MLFuncs.avg(currScores));
//			}
//		}		
//		return avgScores;		
//	}
	
	//////////////////////////////////////////
	// Grigori: These methods are never used
	//////////////////////////////////////////
//	
//	public double getRMSError() {
//		return rmsError;
//	}
//
//	public Matrix getEvalMat() {
//		return cvaEvalMat;
//	}
//
//	public Matrix getEvects() {
//		return cvaEvectsSmall;
//	}
//
//	public Matrix getChiSqrInfo() {
//		return chiSqrInfo;
//	}
//
//	public double[][] getConfidReg() {
//		return confidReg;
//	}
//
//	public int getNumSigCVDim() {
//		return nSigDim;
//	}

//	/**
//	 * Returns 2D array of confidence regions for each class ('group') and cv
//	 * dimension
//	 * 
//	 * @param chiSqrProbs -
//	 *            double array of prob vals for each dim
//	 * @param thresh
//	 * @return
//	 * @throws NpairsjException
//	 */
//	private double[][] confidReg(double[] chiSqrProbs, double thresh)
//			throws NpairsjException, MathException {
//
//		double[][] confidReg = new double[nClasses][nCVDim];
//		int numPAboveThresh = 0;
//		for (double p : chiSqrProbs) {
//			if (p > thresh) {
//				++numPAboveThresh;
//			}
//		}
//
//		if (numPAboveThresh > 0) {
//			double sumSqrdEv = 0;
//			for (int i = 0; i < nClasses; ++i) {
//				for (int j = 0; j < nCVDim; ++j) {
//					// compute scaling factor for non-spherical variances
//					for (int k = 0; k < cvaScores.numRows(); ++k) {
//						sumSqrdEv += cvaScores.get(k, j) * cvaScores.get(k, j);
//					}
//					double sFact = sumSqrdEv
//							/ ((1 + cvaEvals[j]) * (nVol - nClasses));
//					if (sFact == 0) {
//						throw new NpairsjException(
//								"Error - cva scores cannot be all zeroes");
//					}
//					sFact = 1 / Math.sqrt(sFact);
//
//					// compute radius for confidence circle
//					confidReg[i][j] = (1 / sFact)
//							* Math.sqrt((chiSqrCutoffVal(thresh,
//									numPAboveThresh))
//									/ clsSz[i]);
//					sumSqrdEv = 0;
//				}
//			}
//		}
//		nSigDim = numPAboveThresh;
//		
//		return confidReg;
//	}
	
//	/**
//	 * calculates chi squared data for each CV dimension (+1?) using algorithm
//	 * of Jon Anderson's IDL npairs code 'cva_sig_dims.pro'.
//	 * 
//	 * @param thresh
//	 * @return Matrix chiSqrInfo 3 row X (num CV dims + 1) col. chiSqrInfo(0,i) =
//	 *         chi squared value (Bartlett's statistic) for ith cv dimension
//	 *         chiSqrInfo(1,i) = prob associated with corresponding chi squared
//	 *         value chiSqrInfo(2,i) = degrees of freedom
//	 */
//
//	// TODO: figure out why there is one more dim than no. CV Dims and how to
//	// handle
//	// it - see CVA.negateAndPermuteDims(int[], int[])
//	private Matrix chiSqrInfo(double thresh) throws MathException {
//
//		int nPC = cvaEvals.length;
//
//		// compute D statistic
//		double[][] chiSqrInfo = new double[3][nCVDim + 1];
//		for (int i = 0; i <= nCVDim; ++i) {
//			for (int j = i + 1; j <= nPC; ++j) {
//				chiSqrInfo[0][i] += Math.log(1 + cvaEvals[j - 1]);
//			}
//			chiSqrInfo[0][i] = (nVol - 1 - ((nPC + nClasses) / 2.0))
//					* chiSqrInfo[0][i];
//			chiSqrInfo[2][i] = (nPC - i) * (nClasses - i - 1);
//			if (chiSqrInfo[2][i] > 0) {
//				chiSqrInfo[1][i] = chiSqrProb(chiSqrInfo[0][i],
//						chiSqrInfo[2][i]);
//			}
//		}
//
//		MatrixImpl mImpl = new MatrixImpl(chiSqrInfo);
//		return mImpl.getMatrix();
//	}
	
		
//	/**
//	 * Returns probability of observing 'val' or something smaller from a
//	 * chi-squared distribution with 'df' degrees of freedom: Probability(X <
//	 * val) where X == chi-squared dist. with 'df' deg. of freedom
//	 * 
//	 * @param val
//	 * @param df
//	 * @return prob
//	 */
//	private static double chiSqrProb(double val, double df)
//			throws MathException {
//		ChiSquaredDistributionImpl chiSqrDist = new ChiSquaredDistributionImpl(
//				df);
//		return chiSqrDist.cumulativeProbability(val);
//
//	}

//	/**
//	 * Returns the cutoff value v such that
//	 * 
//	 * Probability(X < v) = a
//	 * 
//	 * where X is a random variable from the chi_sqr distribution with v degrees
//	 * of freedom.
//	 * 
//	 * 
//	 * @param prob
//	 * @param df
//	 * @return cutoff value
//	 */
//	private static double chiSqrCutoffVal(double prob, double df)
//			throws MathException {
//		ChiSquaredDistributionImpl chiSqrDist = new ChiSquaredDistributionImpl(
//				df);
//		return chiSqrDist.inverseCumulativeProbability(prob);
//	}
//	/**
//	 * Returns rms (root mean-squared) error of |MV - VL|, where M is W^(-1) *
//	 * B, V is evects, L is evals. REQUIRED: Matrix M must be square.
//	 * 
//	 * @param M
//	 *            data matrix
//	 * @param evects
//	 *            V matrix
//	 * @param evals
//	 *            L matrix
//	 * @return rms error
//	 */
//	private static double rmsError(Matrix M, Matrix evects, Matrix evals) {
//		double rmsErr = 0;
//		int numDims = M.numCols();
//
//		// calculate MV - VL (result lives in diffMat)
//		Matrix diffMat = M.mult(evects);
//		diffMat = diffMat.plusEquals(evects.mult(evals).mult(-1));
//
//		for (int dim = 0; dim < numDims; ++dim) {
//			double sumOfSqrdDiffs = 0;
//			for (int row = 0; row < numDims; ++row) {
//				sumOfSqrdDiffs += Math.pow(diffMat.getQuick(row, dim), 2);
//			}
//			rmsErr += Math.sqrt(sumOfSqrdDiffs / numDims);
//		}
//		// get avg rmsErr over all dims
//		rmsErr = rmsErr / numDims;
//		return rmsErr;
//	}
	
//	/** Returns within-class sums-of-squares-and-products (SSP) Matrix 
//	 *  (actually SUM of individual class SSP Matrices - NOTE that this 
//	 *  is equivalent in variance structure to the *average* group SSP mat))
//	 * @param data
//	 * @return W - within-class SSP Matrix
//	 */
//	protected Matrix getW(Matrix data, int[] classLabels) throws NpairsjException {
//		
//		Hashtable<Integer, int[]> clsIndices = getLabelIndices(classLabels);		
//        //		 TODO: Using pls MLFuncs code here but not currently in
//		//       getLabelIndices for same function 
//		int[] sortedUniqClasses = MLFuncs.sortAscending(MLFuncs
//				.unique(classLabels));
//		nClasses = sortedUniqClasses.length;
//		clsSz = new int[nClasses];
//	    int nCols = data.numCols();
//		
//		// calculate W
//		Matrix W;
//		try {
//			W = (new MatrixImpl(nCols, nCols).getMatrix());
//
//			for (int g = 0; g < nClasses; ++g) {
//				int[] currClsIndices = clsIndices.get(sortedUniqClasses[g]);
//				clsSz[g] = currClsIndices.length;
//				if (clsSz[g] < 2) {
//					throw new NpairsjException(
//					"Each class must contain at least 2 samples");
//				}
//
//				// compute SSP mat for current grp and add it to W
//				Matrix currClsData = data.subMatrixRows(currClsIndices);
//				currClsData = currClsData.meanCentreColumns();
//
//				Matrix currSSPMat = currClsData.sspByCol();
//                W = W.plusEquals(currSSPMat);
//			}	
//		} 
//		catch (NullPointerException e) {
//			throw new NpairsjException(e.getMessage());
//		}
//		
//		return W;
//	}

	
//	/** Returns between-class SSP Matrix B
//	 *  (B = T - W where T == total SSP Matrix and
//	 *   W == within-class SSP Matrix)
//	 * NOTE: input data Matrix is modified! (Columns are mean centred)
//	 * @param data
//	 * @param W
//	 * @return B
//	 */
//	protected static Matrix getB(Matrix data, Matrix W) {
//		
//		data = data.meanCentreColumns(); 
//
////		double sTimeT = System.currentTimeMillis();
//		Matrix T = data.sspByCol();
////		if (debug) {
////			double tTimeT = (System.currentTimeMillis() - sTimeT) / 1000;
////			System.out.println("Time calculating SSP of CVA input data: "
////					+ tTimeT + " s");
////		}
//
//		// calculate B (between-groups SSP matrix) = T - W
//		Matrix B = T.copy();
//		B = B.plusEquals(W.mult(-1));
//		
//		return B;
//	}

//	private Matrix getInvWTimesB() {
//		return invWTimesB;
//	}
//	/**
//	 * Performs CVA on input data. Input data is modified (col-mean-centred).
//	 * 
//	 * @param data
//	 * @throws NpairsjException
//	 */
//	private void computeCVA(Matrix data) throws NpairsjException {
//	
//		Matrix W = getW(data, classLabels);
//		Matrix B = getB(data, W);
//
//
//		// calculate W^(-1) * B
//		double sTime = System.currentTimeMillis();
//		Matrix invWTimesB = W.inverse().mult(B);
//		double tTime = (System.currentTimeMillis() - sTime) / 1000;
//		if (debug) {
//			System.out.println("Time to invert W: " + tTime + " s");
//		}
//
//		// calculate CVA eigenvalues/eigenvectors
//		sTime = System.currentTimeMillis();
//		EigenvalueDecomposition evDecomp = invWTimesB.eigenvalueDecomposition();
//		// NOTE: eigenvalues of W^(-1) * B are real and >= 0, so can ignore
//		//       imaginary part
//		if (debug) {
//			tTime = (System.currentTimeMillis() - sTime) / 1000;
//			System.out.println("Time evd of W^-1B: " + tTime + " s");
//		}
//		nCVDim = Math.min(nClasses - 1, data.numCols());
//		cvaEvalMat = evDecomp.getRealEvalMat();
//		cvaEvectsSmall = evDecomp.getEvects();
//
//		// Calculate RMS error of |MV - VL|, where M is W^(-1) * B,
//		// V is evects, L is evals
//		rmsError = rmsError(invWTimesB, cvaEvectsSmall, cvaEvalMat);
//
//		cvaEvals = evDecomp.getRealEvals();
//		// cvaEvals = evDecomp.getRealEvals(nCVDim);
//
//		// trim evals/evects Matrices
//		cvaEvalMat = evDecomp.getRealEvalMat(nCVDim);
//		cvaEvectsSmall = evDecomp.getEvects(nCVDim);
//
//		// normalize eigenvectors so each has length 1
//		// for (int i = 0; i < cvaEvects.numCols(); ++i) {
//		// double[] currEvect = cvaEvects.getColumn(i);
//		// double sumOfSq = 0.0;
//		//		
//		// for (double val : currEvect) {
//		// sumOfSq += (val * val);
//		//						
//		// }
//		//					
//		// for (int j = 0; j < currEvect.length; ++j) {
//		// currEvect[j] = currEvect[j] / Math.sqrt(sumOfSq);
//		// }
//		// cvaEvects.setColumn(i, currEvect);
//		//						
//		// }
//
//		// normalize eigenvectors so each group has variance 1
//		int nObs = classLabels.length;
//		Matrix scaledW = W.mult(1. / (nObs - nClasses));
//
//		for (int i = 0; i < cvaEvectsSmall.numCols(); ++i) {
//			Matrix currEvect = cvaEvectsSmall.subMatrixCols(new int[] { i });
//			double d = currEvect.transpose().mult(scaledW).mult(currEvect)
//			.getQuick(0, 0);
//			if (d > 0) {
//				currEvect = currEvect.mult(1 / Math.sqrt(d));
//				cvaEvectsSmall.setColumnQuick(i, currEvect.getColumnQuick(0));
//			} 
//			else {
//				for (int j = 0; j < cvaEvectsSmall.numRows(); ++j) {
//					cvaEvectsSmall.setQuick(j, i, 0);
//				}
//			}
//		}
//
//		// calculate canonical scores
//		cvaScores = data.mult(cvaEvectsSmall);
//
//	}
//	
}
