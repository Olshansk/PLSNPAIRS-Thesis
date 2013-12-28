package npairs.utils;

import npairs.shared.matlib.*;

import java.util.*;

import npairs.*;
import npairs.io.*;

import org.apache.commons.math.MathException;
import org.apache.commons.math.distribution.ChiSquaredDistributionImpl;
import pls.shared.MLFuncs;
import java.io.IOException;

/**
 * Performs Canonical Variates Analysis (CVA) on input data Matrix, given input
 * class labels for rows of data Matrix. Calculates eigenvectors and eigenvalues
 * of W^(-1) * B, where W is the within-class covariance matrix and B is the
 * between-class covariance matrix. Also calculates rms (root mean-squared)
 * error of |MV - VL|, where M is W^(-1) * B, V is evects, L is evals, to
 * determine goodness of fit of eigenvalue/vector estimates.
 * 
 * Input data Matrix is modified (col-mean-centred).
 * 
 * REQUIRED: npairs.shared.matlib.Matrix_Impl.matlibType is set outside of this
 * class.
 * 
 * @author anita
 * 
 */
public class CVA {

	private boolean debug = false;

	protected double rmsError;

	protected Matrix cvaEvalMat;

	protected double[] cvaEvals;

	protected Matrix cvaEvectsSmall;	// cva eigenimages before projection back into any other basis

	protected Matrix cvaScores;   // contains projection of input data onto cv
								// eigenimages,
								// i.e., representation of input data in cv space;
								// num rows = num rows in input data;
								// num cols = num CV dims
	protected Matrix cvaEigimsBig = null; // cva eigenimages projected back into
	                                 // 'original' space (if init. feature selection
	                                 // step was done before second-level basis
	                                 // decomposition step, this data will have
	                                 // been projected back through 2 transformations
	                                 // to get back into original space)
	
	protected Matrix cvaFeatSpEigims = null; // cva eigenimages projected back once, through basis 
	                          // decomposition space (e.g. PCA), but not through
	                          // init. feature selection step. 
	

	protected Matrix chiSqrInfo;

//	private double[][] confidReg;

	protected int nSigDim;
	
	private Matrix r2; // contains r^2 values calculated for each input data dim and CV dim
	                   // size(r2) = no. input data dims (e.g. no. PC dims) rows X 
	                   // no. CV dims cols

	protected int[] classLabels;

	protected int nCVDim;

	protected int nClasses;

	protected int[] clsSz; // contains class sizes for each class label (class labels
						 // in ascending order)

	protected int nVol;
	
	private boolean computeR2 = false; // Recommended: set to false unless you want 
	                                   // to examine R2 output (since it's a non-trivial
	                                   // exercise to compute R2 values).
	private boolean computeChiSqr = false; // recommended: set to false unless
	                                        // you want to examine chi squared output 
	                                        // (since it's non-trivial to calculate)
	protected boolean cvaInOrigSpace; // true if cva eigims live in original (image) space
	
//	private static int count = 0; // FOR DEBUGGING

	
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
	 * @param computeAllStats indicates whether to calculate R2 and ChiSqr stats
	 * @param dataInOrigSpace true if input data lives in original (image) space 
	 *                       instead of projected into e.g. PCA or some other feature 
	 *                       selection space
	 * @see createEigenimages()
	 * @see rotateEigimsToOrigSpace()
	 * 
	 * 
	 */
	public CVA(Matrix data, int[] labels, boolean computeAllStats, boolean dataInOrigSpace) throws NpairsjException {
		this.computeR2 = computeAllStats;
		this.computeChiSqr = computeAllStats;
	
		if (debug) {
			System.out.println("CVA input labels (length = " + labels.length + "): ");
			NpairsjIO.print(labels);
			System.out.println("Size input data: " + data.numRows() + " X " + data.numCols());
		}
			
		if (data.numRows() != labels.length) {
			throw new IllegalArgumentException(
					"No. rows in input Matrix does not match "
							+ "length of input array");
		}

		final double CHISQR_THRESH = 0.95;
		classLabels = labels;
		nVol = classLabels.length;
		
//		double sTime = System.currentTimeMillis();
		computeCVA(data);
		this.cvaInOrigSpace = dataInOrigSpace;
//		if (debug) {
//			double tTime = (System.currentTimeMillis() - sTime) / 1000;
//			System.out.println("Total time CVA: " + tTime + " s");
//		}
		
		r2 = computeR2(data);
		
		if (computeChiSqr) {

			try {

				Npairsj.output.print("\tCalculating chi sqr...");
				double sTime = System.currentTimeMillis();
				chiSqrInfo = chiSqrInfo(CHISQR_THRESH);
				double tTime = (System.currentTimeMillis() - sTime) / 1000;
				Npairsj.output.println("[" + tTime + " s]");

				//				System.out.println("Calculating confidence regions...");
				//				sTime = System.currentTimeMillis();
				//			confidReg = confidReg(chiSqrInfo.getRow(1), CHISQR_THRESH);
				//			if (debug) {
				//				double tTime = (System.currentTimeMillis() - sTime) / 1000;
				//				System.out.println("Finished calculating conf. reg.: " + tTime);
				//			}
			} 
			catch (MathException e) {
				throw new NpairsjException("Problem encountered calculating chi squared info for CVA: " 
						+ e.getMessage());
			}
		}
	}

	private Matrix computeR2(Matrix data) {
		// for each CV dim, find correlation coefficients between CV scores and 
		// 'timecourse' for each input data dim (e.g. PC or voxel dim) 
		return data.correlate(cvaScores).squareElements();
	
	}

	// for testing
	public CVA() {};
	
	/**
	 * Performs CVA on input data. Input data is modified (col-mean-centred).
	 * 
	 * @param data
	 * @throws NpairsjException
	 */
	private void computeCVA(Matrix data) throws NpairsjException {
	
		Matrix W = getW(data, classLabels);
//		if (debug) {

			EigenvalueDecomposition evd = W.eigenvalueDecomposition();
			double[] evals = evd.getRealEvals();
			
			double cond = MLFuncs.max(evals) / MLFuncs.min(evals);
			
			if (debug) {
				Npairsj.output.print("(W cond. no. =  ");
				Npairsj.output.printf("%.3f) ", cond);
			}
			if (Math.abs(cond) > 1000) {
				Npairsj.output.printf("\nt*WARNING: W is nearly " 
						+ "singular! (W cond. no. = %.3f)* ", cond);
			}
//		}
		
		Matrix B = getB(data, W);


		// calculate W^(-1) * B

		Matrix invWTimesB = null;
		try { 
			invWTimesB = W.inverse().mult(B);
		} catch (IllegalArgumentException e) {
			throw new NpairsjException("Within-class covariance matrix W is singular! " +
					"\nMaybe you included too many PC dimensions in the CVA analysis." +
					"\nSee analysis log file in results directory for more details.");
		}

		// calculate CVA eigenvalues/eigenvectors

		EigenvalueDecomposition evDecomp = invWTimesB.eigenvalueDecomposition();
		// TODO: verify that eigenvalues of W^(-1) * B are real and >= 0, 
		// so can ignore imaginary part
		
//		if (debug) {
//			tTime = (System.currentTimeMillis() - sTime) / 1000;
//			System.out.println("Time EVD of W^(-1)*B: " + tTime + " s");
//		}
		nCVDim = Math.min(nClasses - 1, data.numCols());
		cvaEvalMat = evDecomp.getRealEvalMat();
		cvaEvectsSmall = evDecomp.getEvects();

		// Calculate RMS error of |MV - VL|, where M is W^(-1) * B,
		// V is evects, L is evals
		rmsError = rmsError(invWTimesB, cvaEvectsSmall, cvaEvalMat);

		// trim evals/evects 
		cvaEvals = evDecomp.getRealEvals(nCVDim);
		cvaEvalMat = evDecomp.getRealEvalMat(nCVDim);
		cvaEvectsSmall = evDecomp.getEvects(nCVDim);

		// normalize eigenvectors so each group (class) has variance 1
		// (compare this.normEvectsByLength())
		normEvectsByVar(W);

		// calculate canonical scores
		cvaScores = data.mult(cvaEvectsSmall);

	}

    /**	 Normalize eigenvectors so each group has variance 1
     *   (i.e. Given evect matrix E, want transpose(E)(W/(n-g))E = I,
     *    where n = no. observations (tmpts) and g = no. classes;
     *    see e.g. Strother et al Neuroimage 2001)
     * @param W within-class covariance matrix (actually SUM of 
     *          individual class SSP Matrices) 
     * @see normEvectsByLength
     */
	private void normEvectsByVar(Matrix W) {
		
		int nObs = classLabels.length;
		Matrix scaledW = W.mult(1. / (nObs - nClasses));

		for (int i = 0; i < cvaEvectsSmall.numCols(); ++i) {
			Matrix currEvect = cvaEvectsSmall.subMatrixCols(new int[] { i });
			double d = currEvect.transpose().mult(scaledW).mult(currEvect).
				getQuick(0, 0);
			if (d > 0) { 
				currEvect = currEvect.mult(1 / Math.sqrt(d));
				cvaEvectsSmall.setColumnQuick(i, currEvect.getColumnQuick(0));
			} 
			else { 
				// d = 0; set evect to 0.  
				for (int j = 0; j < cvaEvectsSmall.numRows(); ++j) {
					cvaEvectsSmall.setQuick(j, i, 0);
				}
			}
		}
	}

	/**	 Normalize eigenvectors so each has length 1
	 * @see normEvectsByVar
	 *
	 */
	@SuppressWarnings("unused")
	private void normEvectsByLength() {
		
		 for (int i = 0; i < cvaEvectsSmall.numCols(); ++i) {
		 double[] currEvect = cvaEvectsSmall.getColumn(i);
		 double sumOfSq = 0.0;
				
		 for (double val : currEvect) {
		 sumOfSq += (val * val);
								
		 }
							
		 for (int j = 0; j < currEvect.length; ++j) {
		 currEvect[j] = currEvect[j] / Math.sqrt(sumOfSq);
		 }
		 cvaEvectsSmall.setColumn(i, currEvect);
								
		 }
	}

	// TODO: Move getLabelIndices(int[]) out of CVA, since also used in
	// ResultSaver,
	// Resampler and Analysis.
	/**
	 * Returns Hashtable containing index info corresp. to input group labels.
	 * 
	 * @param labels -
	 *            int array of group labels
	 * @return Hashtable<Integer, int[]> where key == group label and value ==
	 *         array of ints containing indices of input array 'labels' corresp.
	 *         to given group label.
	 */
	public static Hashtable<Integer, int[]> getLabelIndices(int[] labels) {
		Hashtable<Integer, Integer> grpSizes = new Hashtable<Integer, Integer>();
		Hashtable<Integer, Vector<Integer>> grpIndices = new Hashtable<Integer, Vector<Integer>>();
		Hashtable<Integer, int[]> grpIndexArrays = new Hashtable<Integer, int[]>();

		// get size and indices of each unique group
		for (int row = 0; row < labels.length; ++row) {
			try {
				grpSizes.put(labels[row], grpSizes.get(labels[row]) + 1);
				grpIndices.get(labels[row]).add(row);
			} catch (NullPointerException e) {
				grpSizes.put(labels[row], 1);
				grpIndices.put(labels[row], new Vector<Integer>());
				grpIndices.get(labels[row]).add(row);
			}
		}

		// stuff grp index info into int[] arrays instead of Vectors
		for (Enumeration<Integer> uniqLabels = grpIndices.keys(); uniqLabels
				.hasMoreElements();) {
			Integer currLabel = (Integer) uniqLabels.nextElement();
			int currGrpSize = grpSizes.get(currLabel).intValue();
			Vector<Integer> currGrpIndexVec = (Vector<Integer>) (grpIndices
					.get(currLabel));
			Integer[] tmpCurrGrpIndices = currGrpIndexVec
					.toArray(new Integer[currGrpSize]);
			int[] currGrpIndices = new int[currGrpSize];
			for (int i = 0; i < currGrpSize; ++i) {
				currGrpIndices[i] = tmpCurrGrpIndices[i].intValue();
			}
			grpIndexArrays.put(currLabel, currGrpIndices);
		}

		return grpIndexArrays;
	}

	/**
	 * Returns rms (root mean-squared) error of |MV - VL|, where M is W^(-1) *
	 * B, V is evects, L is evals. REQUIRED: Matrix M must be square.
	 * 
	 * @param M
	 *            data matrix
	 * @param evects
	 *            V matrix
	 * @param evals
	 *            L matrix
	 * @return rms error
	 */
	private static double rmsError(Matrix M, Matrix evects, Matrix evals) {
		double rmsErr = 0;
		int numDims = M.numCols();

		// calculate MV - VL (result lives in diffMat)
		Matrix diffMat = M.mult(evects);
		diffMat = diffMat.plusEquals(evects.mult(evals).mult(-1));

		for (int dim = 0; dim < numDims; ++dim) {
			double sumOfSqrdDiffs = 0;
			for (int row = 0; row < numDims; ++row) {
				sumOfSqrdDiffs += Math.pow(diffMat.getQuick(row, dim), 2);
			}
			rmsErr += Math.sqrt(sumOfSqrdDiffs / numDims);
		}
		// get avg rmsErr over all dims
		rmsErr = rmsErr / numDims;
		return rmsErr;
	}

	/** Returns within-class sums-of-squares-and-products (SSP) Matrix 
	 *  (actually SUM of individual class SSP Matrices - NOTE that this 
	 *  is equivalent in variance structure to the *average* group SSP mat))
	 * @param data
	 * @return W - within-class SSP Matrix
	 */
	protected Matrix getW(Matrix data, int[] classLabels) throws NpairsjException {
		
		Hashtable<Integer, int[]> clsIndices = getLabelIndices(classLabels);		
        //		 TODO: Using pls MLFuncs code here but not currently in
		//       getLabelIndices for same function 
		int[] sortedUniqClasses = MLFuncs.sortAscending(MLFuncs
				.unique(classLabels));
		nClasses = sortedUniqClasses.length;
		clsSz = new int[nClasses];
	    int nCols = data.numCols();
		
		// calculate W
		Matrix W;
		try {
			W = (new MatrixImpl(nCols, nCols).getMatrix());

			for (int g = 0; g < nClasses; ++g) {
				int[] currClsIndices = clsIndices.get(sortedUniqClasses[g]);
				clsSz[g] = currClsIndices.length;
				if (clsSz[g] < 2) {
					throw new NpairsjException(
					"Each class must contain at least 2 samples");
				}

				// compute SSP mat for current grp and add it to W
				Matrix currClsData = data.subMatrixRows(currClsIndices);
				currClsData = currClsData.meanCentreColumns();

				Matrix currSSPMat = currClsData.sspByCol();
                W = W.plusEquals(currSSPMat);
			}	
		} 
		catch (NullPointerException e) {
			throw new NpairsjException(e.getMessage());
		}
		
		return W;
	}

	
	/** Returns between-class SSP Matrix B
	 *  (B = T - W where T == total SSP Matrix and
	 *   W == within-class SSP Matrix)
	 * NOTE: input data Matrix is modified! (Columns are mean centred)
	 * @param data
	 * @param W
	 * @return B
	 */
	protected static Matrix getB(Matrix data, Matrix W) {
		
		data = data.meanCentreColumns(); 

//		double sTimeT = System.currentTimeMillis();
		Matrix T = data.sspByCol();
//		if (debug) {
//			double tTimeT = (System.currentTimeMillis() - sTimeT) / 1000;
//			System.out.println("Time calculating SSP of CVA input data: "
//					+ tTimeT + " s");
//		}

		// calculate B (between-groups SSP matrix) = T - W
		Matrix B = T.copy();
		B = B.plusEquals(W.mult(-1));
		
		return B;
	}

//	private Matrix getInvWTimesB() {
//		return invWTimesB;
//	}

	public double getRMSError() {
		return rmsError;
	}

	public Matrix getEvalMat() {
		return cvaEvalMat;
	}

	public double[] getEvals() {
		return cvaEvals;
	}

	public Matrix getCVScores() {
		return cvaScores;
	}

	public Matrix getEvects() {
		return cvaEvectsSmall;
	}

	public Matrix getEigimsBig() {
		return cvaEigimsBig;
	}
	
	public Matrix getFeatSpEigims() {
		return cvaFeatSpEigims;
	}

	public Matrix getChiSqrInfo() {
		return chiSqrInfo;
	}

//	public double[][] getConfidReg() {
//		return confidReg;
//	}

	public int getNumSigCVDim() {
		return nSigDim;
	}

	// Set method(s):
//	public void setEigims(Matrix newEigims) {
//		if ((newEigims.numCols() != cvaEigims.numCols())) {
//			throw new IllegalArgumentException(
//					"Input Matrix must have same number "
//							+ "of columns (no. CV Dims) as CVA eigenimages");
//		}
//		cvaEigims = newEigims;
//	}

	/**
	 * Creates CVA eigenimages by projecting CVA eigenvectors onto input basis
	 * vectors (e.g. PCA eigenvectors, if PCA was run on data before passing it
	 * to CVA).
	 * 
	 * @param basisVects -
	 *            Matrix of vectors onto which to project eigenvectors (columns ==
	 *            vectors) - if null, then sets cva eigenimages == cva
	 *            eigenvectors (transposed) (set to null if no PCA was
	 *            performed) NOTE dims of eigenimages will be 
	 *            nDataDims (vox or init feat dims) * nCVDims
	 * @param basisVectsInOrigSpace true if basisVects live in original (image) space
	 *                              false if basisVects themselves live in projected
	 *                              (e.g. initial feature selection) space
	 */
	public void createFeatSpEigenimages(Matrix basisVects, boolean basisVectsInOrigSpace) {
		if (basisVects != null) {
			cvaFeatSpEigims = basisVects.mult(cvaEvectsSmall);
			cvaInOrigSpace = basisVectsInOrigSpace;
		} else {
			cvaFeatSpEigims = cvaEvectsSmall;
		}
	}

	/** Applies full Procrustes transformation proc to CV dims in
	 *  this CVA result.  Fields affected: Eigenvalues, CV scores, 
	 *  CV eigenimages, Chi squared info, R2 info. 
	 * @param proc
	 */
	public void applyFullProcrust(Procrustes proc) {
		
		// CV eigenimages: apply proc Xform
		cvaEvectsSmall = cvaEvectsSmall.mult(proc.getRot().transpose());
		if (cvaFeatSpEigims != null) {
			cvaFeatSpEigims = cvaFeatSpEigims.mult(proc.getRot().transpose());
		}
		if (cvaEigimsBig != null) {
			cvaEigimsBig = cvaEigimsBig.mult(proc.getRot().transpose());
		}
		
		// CV scores: apply proc Xform
		cvaScores = cvaScores.mult(proc.getRot().transpose());
		
		// CV evals, chi squared, R2: ??
		
	}

	/**
	 * Negate and permute CV dims of this CVA result according to negation and
	 * permutation info supplied in input args. Fields affected: Eigenvalues -
	 * permuted; CV scores - negated and permuted; CV eigenimages - negated and
	 * permuted; Chi squared info - permuted; R2 info - permuted
	 * 
	 * @param sign -
	 *            int array consisting of 1's and -1's. - if ith permuted CV dim
	 *            is to be negated, sign[i] = -1; if not, sign[i] = 1.
	 * @param permutation -
	 *            int array indicating new order of CV dims. - e.g. permutation =
	 *            {0, 2, 1}, sign = {-1, 1, -1} means newScores[0] = -1 *
	 *            oldScores[0] newScores[1] = 1 * oldScores[2] newScores[2] = -1 *
	 *            oldScores[1]
	 * 
	 * @throws NpairsjException
	 *             if input args are not of length = no. of CV dims
	 */
	public void negateAndPermuteDims(int[] sign, int[] permutIndex)
			throws NpairsjException {

		if (sign.length != nCVDim || permutIndex.length != nCVDim) {
			throw new NpairsjException(
					"Input arrays must have length = no. of CV dims");
		}

		double[] tmpEvals = new double[nCVDim];
		for (int dim = 0; dim < nCVDim; ++dim) {
			tmpEvals[dim] = cvaEvals[permutIndex[dim]];
		}
		cvaEvals = tmpEvals;

		try {
			cvaEvalMat.setDiag(cvaEvals);
		}
		catch (MatrixException me) {
			// matrix is square by construction
		}
		
		cvaScores = cvaScores.permuteColumns(permutIndex);
		cvaEvectsSmall = cvaEvectsSmall.permuteColumns(permutIndex);
		if (cvaFeatSpEigims != null) {
			cvaFeatSpEigims = cvaFeatSpEigims.permuteColumns(permutIndex);
		}
		if (cvaEigimsBig != null) {
			cvaEigimsBig = cvaEigimsBig.permuteColumns(permutIndex);
		}
		
		if (computeR2) {
			r2 = r2.permuteColumns(permutIndex);
		}

		// TODO: Figure out how to handle extra chi-squared dim
		// (note: in idl cva_reference.pro, extra dim is ignored;
		// permutation is applied as if chi-squared info only has
		// no. CV dims dimensions.

		if (computeChiSqr) {
			int[] chiSqrPermutIndex = new int[nCVDim + 1];
			for (int i = 0; i < nCVDim; ++i) {
				chiSqrPermutIndex[i] = permutIndex[i];
			}
			chiSqrPermutIndex[nCVDim] = nCVDim;

			chiSqrInfo = chiSqrInfo.permuteColumns(chiSqrPermutIndex);
		}

		for (int dim = 0; dim < nCVDim; ++dim) {
			double[] newSignScores = MLFuncs.product(cvaScores.getColumn(dim),
					sign[dim]);
			cvaScores.setColumn(dim, newSignScores);
			double[] newSignEvects = MLFuncs.product(cvaEvectsSmall.getColumn(dim),
					sign[dim]);
			cvaEvectsSmall.setColumn(dim, newSignEvects);
			if (cvaFeatSpEigims != null) {
				double [] newSignEigims = MLFuncs.product(cvaFeatSpEigims
						.getColumn(dim), sign[dim]);
				cvaFeatSpEigims.setColumn(dim, newSignEigims);
			}
			if (cvaEigimsBig != null) {
				double[] newSignEigims = MLFuncs.product(cvaEigimsBig
						.getColumn(dim), sign[dim]);
				cvaEigimsBig.setColumn(dim, newSignEigims);
			}
		}
	}

	/**
	 * calculates chi squared data for each CV dimension (+1?) using algorithm
	 * of Jon Anderson's IDL npairs code 'cva_sig_dims.pro'.
	 * 
	 * @param thresh
	 * @return Matrix chiSqrInfo 3 row X (num CV dims + 1) col. chiSqrInfo(0,i) =
	 *         chi squared value (Bartlett's statistic) for ith cv dimension
	 *         chiSqrInfo(1,i) = prob associated with corresponding chi squared
	 *         value chiSqrInfo(2,i) = degrees of freedom
	 */

	// TODO: figure out why there is one more dim than no. CV Dims and how to
	// handle
	// it - see CVA.negateAndPermuteDims(int[], int[])
	private Matrix chiSqrInfo(double thresh) throws MathException {

		int nPC = cvaEvals.length;

		// compute D statistic
		double[][] chiSqrInfo = new double[3][nCVDim + 1];
		for (int i = 0; i <= nCVDim; ++i) {
			for (int j = i + 1; j <= nPC; ++j) {
				chiSqrInfo[0][i] += Math.log(1 + cvaEvals[j - 1]);
			}
			chiSqrInfo[0][i] = (nVol - 1 - ((nPC + nClasses) / 2.0))
					* chiSqrInfo[0][i];
			chiSqrInfo[2][i] = (nPC - i) * (nClasses - i - 1);
			if (chiSqrInfo[2][i] > 0) {
				chiSqrInfo[1][i] = chiSqrProb(chiSqrInfo[0][i],
						chiSqrInfo[2][i], thresh);
			}
		}

		MatrixImpl mImpl = new MatrixImpl(chiSqrInfo);
		return mImpl.getMatrix();
	}

	/**
	 * Returns 2D array of confidence regions for each class ('group') and cv
	 * dimension
	 * 
	 * @param chiSqrProbs -
	 *            double array of prob vals for each dim
	 * @param thresh
	 * @return
	 * @throws NpairsjException
	 */
	private double[][] confidReg(double[] chiSqrProbs, double thresh)
			throws NpairsjException, MathException {

		double[][] confidReg = new double[nClasses][nCVDim];
		int numPAboveThresh = 0;
		for (double p : chiSqrProbs) {
			if (p >= thresh) {
				++numPAboveThresh;
			}
		}

		if (numPAboveThresh > 0) {
			double sumSqrdEv = 0;
			double sFact = 0;
			for (int i = 0; i < nClasses; ++i) {
				for (int j = 0; j < nCVDim; ++j) {
					// compute scaling factor for non-spherical variances
					for (int k = 0; k < cvaScores.numRows(); ++k) {
						sumSqrdEv += cvaScores.getQuick(k, j) * cvaScores.getQuick(k, j);
					}
					sFact = sumSqrdEv
							/ ((1 + cvaEvals[j]) * (nVol - nClasses));
					if (sFact == 0) {
						throw new NpairsjException(
								"Error - cva scores cannot be all zeroes");
					}
					sFact = 1 / Math.sqrt(sFact);

					// compute radius for confidence circle
					confidReg[i][j] = (1 / sFact)
							* Math.sqrt((chiSqrCutoffVal(thresh,
									numPAboveThresh))
									/ clsSz[i]);
					sumSqrdEv = 0;
				}
			}
		}
		nSigDim = numPAboveThresh;
		
		return confidReg;
	}

	/**
	 * Returns probability of observing 'val' or something smaller from a
	 * chi-squared distribution with 'df' degrees of freedom: Probability(X <
	 * val) where X == chi-squared dist. with 'df' deg. of freedom
	 * 
	 * @param val
	 * @param df
	 * @param thresh TODO
	 * @return prob
	 */
	private static double chiSqrProb(double val, double df, double thresh) 
		throws MathException {
		
		ChiSquaredDistributionImpl chiSqrDist = new ChiSquaredDistributionImpl(df);
		double prob = -1;
		try {
			prob =  chiSqrDist.cumulativeProbability(val);
		}
		catch (MathException e) {
			double criticalVal = chiSqrDist.inverseCumulativeProbability(thresh);
//			System.out.println("Critical value for alpha = " + thresh + ": " + criticalVal);
//			System.out.println("Value to be evaluated: " + val);
			if (val > criticalVal) {
				// set probability of val > criticalVal to 1.0 to avoid convergence problems
				// in cumulativeProbability calculation of cdf for high vals.
				prob = 1.0;
			}
			else {
				throw new MathException("(CVA) Problem encountered calculating cumul. prob. of chi squared dist.: " +
					e.getMessage());					
			}
		}
		return prob;
	}

	/**
	 * Returns the cutoff value v such that
	 * 
	 * Probability(X < v) = a
	 * 
	 * where X is a random variable from the chi_sqr distribution with df degrees
	 * of freedom.
	 * 
	 * 
	 * @param prob
	 * @param df
	 * @return cutoff value 
	 */
	private static double chiSqrCutoffVal(double prob, double df)
			throws MathException {
		ChiSquaredDistributionImpl chiSqrDist = new ChiSquaredDistributionImpl(
				df);
		return chiSqrDist.inverseCumulativeProbability(prob);
	}

	/** Project CVA results back into original space, if CVA input was in some 
	 *  other (feature-selection) space.
	 * @param cva
	 * @param projFactorMat
	 * @param origData
	 */
	public void rotateEigimsToOrigSpace(Matrix projFactorMat, Matrix origData) {
		if (cvaInOrigSpace) {
			cvaEigimsBig = cvaFeatSpEigims;
			return;
		}
		// project cva results back onto projFactorMat (== inverted feature-selection 
		// projection Matrix for regular EVD; 
		// TODO: update documentation for unweighted EVD
		// (see PCA.rotateEigimsToOrigSpace(...) documentation for algebraic details)

		Matrix P1invP = cvaFeatSpEigims.transpose().mult(projFactorMat);

		Matrix voxSpaceCVAEigims = P1invP.mult(origData);

		cvaEigimsBig = voxSpaceCVAEigims.transpose();
		cvaInOrigSpace = true;
	}
	
	
	/**
	 * Saves CVA results to file. If original data has been transformed into new
	 * vector space (e.g. via pca), then this method does NOT rotate data back
	 * into the original space first (must do outside of this method).
	 * Note that this method does NOT save r2 data by default.  Must set
	 * boolean saveR2 to true for r2 data to be saved here.
	 * 
	 * @param cvaSavePref
	 *            prefix (including path) of saved CVA files
	 * @param saveAsSplit
	 * @param splitNum
	 * @param splitHalf
	 * @param saveR2 TODO
	 * @param saveR2 
	 * @see #Npairsj.saveR2
	 */
	// TODO: only save input number of CV dims.
	public void saveCVAResultsIDL(String cvaSavePref, boolean saveAsSplit,
			int splitNum, int splitHalf, boolean saveR2) throws IOException {
		String cvaEvalFile;
		String saveFormat = "IDL";
		String cvaEigimFile;
		String cvaScoreFile;
		String cvaRMSFile;
		String cvaClassInfoFile;
		String cvaChiSqrInfoFile;
		String cvaConfidRegFile;
		String cvaEvectFile;
		String cvaR2File = null;

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
			if (saveR2) {
				cvaR2File = cvaSavePref + ".CVA." + splitNum + "."
					+ splitHalf + ".r2";
			}
		} else {
			cvaEvalFile = cvaSavePref + ".CVA.ALL.eigval";
			cvaEigimFile = cvaSavePref + ".CVA.ALL.eigim";
			cvaScoreFile = cvaSavePref + ".CVA.ALL.can";
			cvaRMSFile = cvaSavePref + ".CVA.ALL.rms";
			cvaClassInfoFile = cvaSavePref + ".CVA.ALL.group";
			cvaChiSqrInfoFile = cvaSavePref + ".CVA.ALL.chi";	
			cvaConfidRegFile = cvaSavePref + ".CVA.ALL.confidReg";
			cvaEvectFile = cvaSavePref + ".CVA.ALL.evect";
			if (saveR2) {
				cvaR2File = cvaSavePref + ".CVA.ALL.r2";
			}
		}

		NpairsjIO.printToIDLFile(cvaEvals, cvaEvalFile);

		cvaEigimsBig.printToFile(cvaEigimFile, saveFormat);
		cvaScores.printToFile(cvaScoreFile, saveFormat);
		double[] rmsErrArray = { rmsError };
		NpairsjIO.printToIDLFile(rmsErrArray, cvaRMSFile);
		NpairsjIO.printRowToIDLFile(classLabels, cvaClassInfoFile);
		if (computeChiSqr) {
			chiSqrInfo.printToFile(cvaChiSqrInfoFile, "IDL");
		}
		// (note: confidreg info not normal idl npairs output)
		//MatrixImpl mImpl = new MatrixImpl(confidReg);
		//mImpl.getMatrix().printToFile(cvaConfidRegFile, "IDL");
		cvaEvectsSmall.printToFile(cvaEvectFile, saveFormat);
		if (saveR2) {
			r2.printToFile(cvaR2File, "IDL");
		}
	}

	public int getNumCVDims() {
		return nCVDim;
	}
	
	/** Returns average CV Scores for each condition (class label)
	 * @return Matrix containing avg cv scores for each condition (class label)
	 *                    - dims [num conds][num cv dims]
	 * @see ResultSaver.avgCVScores(double[][], int[], int[], int[])
	 */
	public Matrix avgCVScores() {
			
		Hashtable<Integer, int[]> condIndices = CVA.getLabelIndices(classLabels);
		int numCond = condIndices.size();
		int numCVDims = cvaScores.numCols();
		int[] sortedUniqCondLabels = new int[numCond];
		int i = 0;
		for (Enumeration<Integer> uniqCondLabels = condIndices.keys(); uniqCondLabels.hasMoreElements(); ) {
			sortedUniqCondLabels[i] = (Integer)uniqCondLabels.nextElement();
			++i;
		}
		sortedUniqCondLabels = MLFuncs.sortAscending(sortedUniqCondLabels);
		
		MatrixImpl avgScoresImpl = new MatrixImpl(numCond, numCVDims);
		Matrix avgScores = avgScoresImpl.getMatrix();
		for (int cond = 0; cond < numCond; ++cond) {
			int[] currCondIndices = condIndices.get(sortedUniqCondLabels[cond]);
			for (int dim = 0; dim < numCVDims; ++dim) {
				double[] currScores = MLFuncs.getItemsAtIndices(cvaScores.getColumn(dim),
						currCondIndices);
				avgScores.set(cond, dim, MLFuncs.avg(currScores));
			}
		}
		
		return avgScores;
	}
	
	/** Returns new Matrix containing test CV Scores
	 * @param testData 
	 * 
	 */
	public Matrix calcTestCVScores(Matrix testData) {
		return testData.mult(cvaFeatSpEigims);
	}

	public Matrix getR2() {
		return r2;
		
	}

	/** ******************************************************************************* */

	// Quick test of CVA.
	// (REQUIRED: input arg 'colt' or 'matlab' to indicate
	// matrix lib type)
//	public static void main(String[] args) {
//		String matlibType = "MATLAB";
//		MatrixImpl.setMatlibType(matlibType);
//		System.out.println("Matlib type: " + matlibType);
//
//		// using a 2-class, 2D dataset with 36 samples, 18 from each class
//		// (Mardia p. 329)
//		double[] x1 = { 191, 185, 200, 173, 171, 160, 188, 186, 174, 163, 190,
//				174, 201, 190, 182, 184, 177, 178, 186, 211, 201, 242, 184,
//				211, 217, 223, 208, 199, 211, 218, 203, 192, 195, 211, 187, 192 };
//		double[] x2 = { 131, 134, 137, 127, 118, 118, 134, 139, 131, 115, 143,
//				131, 130, 133, 130, 131, 127, 126, 107, 122, 114, 131, 108,
//				118, 122, 127, 125, 124, 129, 126, 122, 116, 123, 122, 123, 109 };
//		double[][] data = new double[2][36];
//		data[0] = x1;
//		data[1] = x2;
//		Matrix transM = new MatrixImpl(data).getMatrix();
//		Matrix M = null;
//		// JMatLink eng = null;
//		String idlFormatMatrixFname = "C:\\anita\\workspace\\PLSNPAIRSGoogleRepo"
//				+ "\\localTestData\\CVATest";
//
//		M = transM.transpose();
//
//		System.out.println("Input data: ");
//		M.print();
//		System.out.println("Saving matrix to file " + idlFormatMatrixFname);
//		M.printToFile(idlFormatMatrixFname, "IDL");
//		int[] labels = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
//				2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 };
//
//		try {
//			CVA cva = new CVA(M, labels);
//			System.out.println("W mat: ");
//			cva.getW().print();
//			System.out.println("B mat: ");
//			cva.getB().print();
//			System.out.println("W^(-1) * B:");
//			cva.getInvWTimesB().print();
//			System.out.println("labels: ");
//			for (int i = 0; i < labels.length; ++i) {
//				System.out.print(labels[i] + " ");
//
//			}
//			System.out.println();
//			System.out.println("Cva evals: ");
//			cva.getEvalMat().print();
//			System.out.println("Cva evects: ");
//			cva.getEvects().print();
//
//			System.out.println("RMS error: " + cva.getRMSError());
//
//			// String cvaSavePref = args[2];
//			// System.out.println("Saving CVA data to file " + cvaSavePref + "
//			// using saveCVAResults...");
//			//
//			// cva.saveCVAResultsIDL(cvaSavePref, false, 0, 1);
//		} catch (NpairsjException e) {
//			e.printStackTrace();
//		}
//		// catch (IOException e) {
//		// e.printStackTrace();
//		// }
//
//	}

	// testing chisqr/confidreg output:

//	private CVA(double[][] cvaScores, double[] cvaEvals, double[] classLabels) {
//		// set class variables required for chisqr/confidreg calculations
//		this.nVol = classLabels.length;
//		this.classLabels = new int[nVol];
//		for (int i = 0; i < nVol; ++i) {
//			this.classLabels[i] = (int) classLabels[i];
//		}
//
//		MatrixImpl mImpl = new MatrixImpl(cvaScores);
//		this.cvaScores = mImpl.getMatrix();
//		this.cvaEvals = cvaEvals;
//
//		this.nCVDim = this.cvaScores.numCols();
//		Hashtable<Integer, int[]> grpIndices = getLabelIndices(this.classLabels);
//		this.nGrp = grpIndices.size();
//		int[] sortedUniqGrpLabels = MLFuncs.sortAscending(MLFuncs
//				.unique(this.classLabels));
//		this.grpSz = new int[nGrp];
//		for (int g = 0; g < nGrp; ++g) {
//			grpSz[g] = grpIndices.get(sortedUniqGrpLabels[g]).length;
//		}
//	}

//	public static void main2(String[] args) {
//
//		MatrixImpl.setMatlibType("COLT");
//		String testNpairsPrefix = "C:\\anita\\workspace\\PLSwithNPAIRS\\localTestData\\testy2r4RasOrtho\\testy2r4RasOrtho";
//		try {
//			double[][] cvaScores = NpairsjIO.readFromIDLFile(testNpairsPrefix
//					+ ".CVA.ALL.can");
//			double[][] cvaEvals2D = NpairsjIO.readFromIDLFile(testNpairsPrefix
//					+ ".CVA.ALL.eigval");
//			double[][] classLabels2D = NpairsjIO
//					.readFromIDLFile(testNpairsPrefix + ".CVA.ALL.group");
//
//			double[] cvaEvals1D = cvaEvals2D[0];
//			double[] classLabels1D = classLabels2D[0];
//
//			// System.out.println("cvaEvals1D size: " + cvaEvals1D.length);
//			// System.out.println("classLabels1D size: " +
//			// classLabels1D.length);
//			System.out.println("Creating test CVA object...");
//			CVA cva = new CVA(cvaScores, cvaEvals1D, classLabels1D);
//			System.out.println("Testing chiSqrInfo...");
//			try {
//				Matrix chiSqrInfo = cva.chiSqrInfo(0.95);
//				System.out.println("Chi squared info: ");
//				chiSqrInfo.print();
//
//				double[][] confidReg = cva
//						.confidReg(chiSqrInfo.getRow(1), 0.95);
//				System.out.println("Confid regs: ");
//				NpairsjIO.print(confidReg);
//			} catch (MathException e) {
//				e.printStackTrace();
//			}
//
//			double val = 0.5;
//			double df = 2;
//
//			double chiSqrProb = chiSqrProb(val, df);
//			System.out.println("prob (X < " + val + "), df = " + df + ": "
//					+ chiSqrProb);
//
//			double chiSqrCutoffVal = chiSqrCutoffVal(chiSqrProb, df);
//			System.out.println("cutoff val s.t. prob(X < val) = " + chiSqrProb
//					+ "; df = " + df + ": " + chiSqrCutoffVal);
//		}
//
//		catch (MathException e) {
//			e.printStackTrace();
//		} catch (NpairsjException e) {
//			e.printStackTrace();
//		} catch (IOException e) {
//			e.printStackTrace();
//		}
//	}
}
