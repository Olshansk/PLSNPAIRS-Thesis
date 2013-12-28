package npairs.utils;

import java.io.Serializable;
import java.io.IOException;

import npairs.io.NpairsDataLoader;
import npairs.shared.matlib.*;
import extern.niftijlib.Nifti1Dataset;

import pls.shared.MLFuncs;
import npairs.Npairsj;
import npairs.NpairsjException;
import npairs.NpairsjSetupParams;

/** Runs PCA on input matlib.Matrix M via EigenvalueDecomposition.
 * 
 * PCA(M) = Eigenvalue decomposition of covariance matrix 
 * of M (i.e., evd(covmat) = V(S^2)Vt, where S^2 is
 * diagonal eigenvalue matrix and covmat = MtM/M.numRows() (after mean-centring M)
 * Note that input M is NOT modified, though, when mean-centring. 
 * (eval = squared singular value lambda in S matrix of SVD(M) = USVt)
 * 
 * given M is m X n,
 * V = PCA eigenvectors (n evects length n in columns of V)
 * S = n X n diagonal matrix containing PCA eigenvalues down diagonal  
 * PC Scores = projection of (mean-centred) original input data 
 * onto PCA eigenvectors V = representation of data in PCA basis
 * 
 * @author anita
 * @version 1.0
 * 
 */
public class PCA implements Serializable{
	
//		private boolean normalizeBySD = false; // if true, PC Scores are normalized by
//											   // their standard deviations
//		                                       // to have variance 1.
		private boolean debug = false;
		private Matrix eigenvectors;  // contains PCA eigenvectors V in columns
		private double[] eigenvalues; // array containing PCA eigenvalues in descending
		                              // order (contents of main diagonal in S)
		private Matrix evalMat;       // diagonal Matrix containing eigenvalues down main diagonal
//		private Matrix covMat; // for debugging
		private Matrix pcScores; // projection of original data onto PC eigenvectors
		                         // (pcScores = MV)
		private static int count = 0;
		
		private boolean pcaInOrigSpace; // Usually do PCA on feature-selected data, in which
		                                // case PCA will
		                                // not be in orig space unless explicitly projected back
		                                // into it.  
		
		
	// Constructor
	public PCA(Matrix M, boolean normalizePCsBySD, boolean inputDataInOrigSpace) {
//		double startTime = System.currentTimeMillis();
		pcaInOrigSpace = inputDataInOrigSpace;
		computePCA(M, normalizePCsBySD);
//		double totTime = (System.currentTimeMillis() - startTime) / 1000;
//		if (debug) {
//			Npairsj.output.println("Total time PCA: " + totTime + " s");
//		}
	}
	
	
	/** Calculates PCA of input Matrix M via eigenvalue decomposition
	 * 
	 * @param M - Matrix of doubles. 
	 *            Assumption: rows are observations, cols are variables
	 * @param normalizeBySD TODO
	 */
	private void computePCA(Matrix M, boolean normalizeBySD) {
		++count;
		int nRows = M.numRows();
		int nCols = M.numCols();
		
		double sTime = System.currentTimeMillis();
		Matrix meanCentredM = M.meanCentreColumns();	// M is NOT modified
		double tTime = (System.currentTimeMillis() - sTime) / 1000;
		if (debug) {
			Npairsj.output.println("\tTime mean-centring PCA input data: " + tTime + " s");
		}
		
		if (nRows < nCols) {
			sTime = System.currentTimeMillis();
			Matrix sspMMt = meanCentredM.sspByRow();
			tTime = (System.currentTimeMillis() - sTime) / 1000;
			if (debug) {
				Npairsj.output.println("\tTime SSP of mean-centred input PCA data: " + tTime + " s");
			}
			
			sTime = System.currentTimeMillis();
			EigenvalueDecomposition evd = sspMMt.eigenvalueDecomposition();
			tTime = (System.currentTimeMillis() - sTime) / 1000;
			if (debug) {
				Npairsj.output.println("\tTime EVD of SSP in PCA: " + tTime + " s");
			}
			
			Matrix leftEigenvects = evd.getEvects(); // = U given MMt = US^2Ut
			Matrix invSqrtEvalMat = evd.getInvSqrtRealEvalMat();
			sTime = System.currentTimeMillis();
			eigenvectors = (meanCentredM.transpose()).mult(leftEigenvects.mult(invSqrtEvalMat));
			tTime = (System.currentTimeMillis() - sTime) / 1000;
			if (debug) {	
				Npairsj.output.println("\tTime calculating PCA eigenvectors: " + tTime + "s");
			}
			// eigenvalues are scaled because evd was done on ssp matrix, 
			// not (sample) covariance matrix
			eigenvalues = scale(evd.getRealEvals(), 1./(double)(nRows - 1));
		}
		
		else {
			sTime = System.currentTimeMillis();
			Matrix sspMtM = meanCentredM.sspByCol();
			tTime = (System.currentTimeMillis() - sTime) / 1000;
			if (debug) {
				Npairsj.output.println("\tTime SSP of mean-centred input PCA data: " + tTime + " s");
//				String saveName = "RRSD_tests/javaTest/pcaSSPMtMMat.debug." + count;
//				sspMtM.printToFile(saveName, "IDL");
			}
			sTime = System.currentTimeMillis();
			EigenvalueDecomposition evd = sspMtM.eigenvalueDecomposition();
			tTime = (System.currentTimeMillis() - sTime) / 1000;
			if (debug) {
				Npairsj.output.println("\tTime EVD of SSP in PCA: " + tTime + " s");
			}
			eigenvectors = evd.getEvects(); // == V given MtM = VS^2Vt
            // eigenvalues are scaled because evd was done on ssp matrix, 
			// not (sample) covariance matrix
			eigenvalues = scale(evd.getRealEvals(), 1./(double)(nRows - 1));
		}		
		
		// TODO: optimize calc. of pc scores.
		sTime = System.currentTimeMillis();
		pcScores = meanCentredM.mult(eigenvectors);
		tTime = (System.currentTimeMillis() - sTime) / 1000;
		if (debug) {
			Npairsj.output.println("\tTime calculating PC scores: " + tTime + " s");
		}
		if (normalizeBySD) {
			sTime = System.currentTimeMillis();
			// normalize each pcScore column (dim) to have variance 1
			// TODO: should we just divide by sqrt(eval) instead of calculating
			// stddev explicitly?  
			for (int d = 0; d < pcScores.numCols(); ++d) {
				double[] currPCs = pcScores.getColumnQuick(d);
				double stdDev = MLFuncs.std(currPCs);
				pcScores.setColumnQuick(d, MLFuncs.divide(currPCs, stdDev));
			}
			if (debug) {
				tTime = (System.currentTimeMillis() - sTime) / 1000;
				Npairsj.output.println("\tTotal time normalizing PC scores: " + tTime + " s");
			}
		}
		evalMat = new MatrixImpl(eigenvalues.length, eigenvalues.length).getMatrix();
		try {
			evalMat.setDiag(eigenvalues);
		}
		catch (MatrixException me) { 
			// matrix is square by construction
		}
	}
	
	
	public Matrix getEvects() {
		return eigenvectors;
	}
	
	public Matrix getEvects(int[] pcDims) {
		return eigenvectors.subMatrixCols(pcDims);
	}
	
	public double[] getEvals() {
		return eigenvalues;
	}
	
	public Matrix getEvalMat() {
		return evalMat;		
	}
	
	/** Returns representation of original data in PC space (i.e., projection
	 *  of original (mean-centred) data onto eigenvectors).  
	 * @return PC Scores, i.e. projection of original data onto PC eigenvectors 
	 */
	public Matrix getPCScores() {
		return pcScores;		
	}
	
	
	/** Returns representation of original data in PC space (i.e., projection
	 * of original (mean-centred) data onto eigenvectors).  Only returns the PC dimensions specified
	 * in input array of PC eigenvector indices-- e.g., if input array is {0,1,4,5},
	 * returns first, second, fifth and sixth PC dims.
	 * @return PC scores, i.e. projection of original data onto subset of 
	 * 			PC eigenvectors
	 */
	public Matrix getPCScores(int[] pcDims) {
		return pcScores.subMatrixCols(pcDims);
	}
	
//	// Set method(s): used in Npairsj.rotateToOrigSpace(PCA)
//	public void setEvects(Matrix newEvects) {
////		if	((newEvects.numCols() != eigenvectors.numCols())) {
////			throw new IllegalArgumentException("Input Matrix and PCA eigenvectors "
////					+ "must have same number of columns (PC Dims)");
////		}
//		eigenvectors = newEvects;
//	}
	
	public boolean pcaInOrigSpace() {
		return pcaInOrigSpace;
		
	}
	
	
	// Truncate pc scores and eigenvalues (and associated evalMat) so only input dims are
	// included.  
	private void truncateEvalsAndScores(int[] pcDims) {
		pcScores = pcScores.subMatrixCols(pcDims);
		eigenvalues = MLFuncs.getItemsAtIndices(eigenvalues, pcDims);
		evalMat = evalMat.subMatrixCols(pcDims);
	
	}
	
	
	/** Saves data in original image space. 
	 *  (Typically called after dimension reduction has been 
	 *  done in PC space, i.e. saves D = U*S*transpose(V*), where * means dim-reduced
	 *  and M = UStranspose(V) is SVD (cf. PCA) decomp. of input data M.)
	 *  REQUIRED: PCA lives in original space (i.e., eigenvectors have been
	 *  rotated back into orig. space if PCA was done in some other space).
	 * @param nsp NpairsjSetupParams containing data/results filenames (and other info)
	 * @param ndl NpairsDataLoader containing mask coords and other data details
	 *  
	 */
	public void saveDataPostPCA(NpairsjSetupParams nsp, NpairsDataLoader ndl) throws NpairsjException,
			IOException {
		double sTime = 0;
		if (debug) {
			sTime = System.currentTimeMillis();
			Npairsj.output.print("\tCreating dim-reduced (denoised) input data...");
		}
		if (!pcaInOrigSpace) {
			throw new NpairsjException("Must rotate PCA eigenvectors into original image " +
					"space before saving denoised (post-PCA) data.");
		}
		
		Matrix dimRedData = pcScores.mult(eigenvectors.transpose());
	
		if (debug) {
			double tTime = (System.currentTimeMillis() - sTime) / 1000;
			Npairsj.output.println("[" + tTime + " s]");
			Npairsj.output.println("\tSize of denoised data matrix: " +
					dimRedData.numRows() + " X " + dimRedData.numCols());
		}

		int[] maskCoords = ndl.getMaskCoords(); // 0-relative
		
		// initialize nifti structure:
		Nifti1Dataset nifti = new Nifti1Dataset();
		// set header filename first and include .nii ext to set dataset file type to .nii
		// (note: also need to write header before writing data when saving .nii file)
		String saveName = nsp.resultsFilePrefix + ".DATA.POST-PCA.nii";
		nifti.setHeaderFilename(saveName);
		nifti.setDataFilename(saveName);
		
		// copy header info from first input data vol (and ndl info, which was also
		// read in from first input data vol) and write it to disk
		Nifti1Dataset vol1Nifti = new Nifti1Dataset(nsp.getDataFilenames()[0]);
		nifti.copyHeader(vol1Nifti);
		nifti.setHeaderFilename(saveName);
		nifti.setDataFilename(saveName);
		nifti.setDatatype((short)64);
		int[] volDims3D = ndl.getDims(); // might as well get 3D dims from ndl 
		int nVols = dimRedData.numRows();
		int xDim = volDims3D[0];
		int yDim = volDims3D[1];
		int zDim = volDims3D[2];
		nifti.setDims((short)4, (short)xDim, (short)yDim, (short)zDim,
				(short)nVols, (short)0, (short)0, (short)0);
		nifti.writeHeader();
		
		// set dimRedData rows into full 3d vols ([Z][Y][X] as per Nifti1Dataset specs)
		double[][][] currVol3D = new double[zDim][yDim][xDim];
		double[] currVol1D = new double[zDim * yDim * xDim];
		
		
		for (short v = 0; v < nVols; ++v) {
			currVol1D = MLFuncs.setVals(currVol1D, maskCoords, dimRedData.getRowQuick(v));
			// turn 1D array into 3D vol
			for (int z = 0; z < zDim; ++z) {
				for (int y = 0; y < yDim; ++y) {
					for (int x = 0; x < xDim; ++x) {
						currVol3D[z][y][x] = currVol1D[z*yDim*xDim + y*xDim + x];
					}
				}
			}
			
			nifti.writeVol(currVol3D, v);
		}
	}
	
	/**If PCA input was not in original data space, this method rotates only the input pca eigenimage 
	 * dimensions back into original space (and discards the rest); 
	 * otherwise would take prohibitive length of time for large data.  
	 * NOTE: pca pc scores and eigenvalues are also truncated to include only input pca dims.
	 * @param pca
	 * @param pcaDims
	 * @param projFactorMat - for regular EVD, this matrix is inverse of projected data matrix 
	 * 							(i.e., inverse of M, the matrix 
	 *                         used as PCA input)
	 *                      - TODO: update documentation explaining projFactorMat for
	 *                        unweighted EVD  
	 * @param origData	data matrix in original space
	 */
	public void rotateEigimsToOrigSpace(int[] pcaDims, Matrix projFactorMat, Matrix origData) {

		if (pcaInOrigSpace) {
			return;
		}
		/** project pca eigenimages back onto inverted feature-selection 
		// projection Matrix, i.e., feat-sel eigenimages (note that inverted
		// proj. matrix == transpose of proj. matrix == Vt)

		// P = MV ==> PVt = M ==> Vt = (invP)M == inverted feat-sel
		// 	proj. Matrix (Vt), i.e. feat-sel eigims == transpose of proj. Matrix V.
		// Vt ( == invSVDEigims, e.g.,) == too large to store, hence
		// store invP = invFeatSelData instead to reconstruct Vt on the fly.
		// Vt = (invP)M ==> PVt = P(invP)M ==> P1Vt = P1(invP)M = M1
		// where P1 = PCA eigenimages in rows, in this case, and M1 = 
		// PCA eigims in rows proj. back into orig. voxel space.
		// given size P1 is reducDataDims X reducDataDims 
		// and size (invP) is reducDataDims X data.numRows(),
		// and size M is data.numRows() X origDataDims,
		// want M1t = origDataDims X reducDataDims();
		// TODO: Want to retain only significant dimensions of PCs! 
		// (see IDL code for comparison)
		 */

		// size P1(invP) is reducDataDims X data.numRows(): 	
//		Matrix P1invP = pca.getEvects(pcaDims).transpose().mult(dataLoader.getInvFeatSelData());
		Matrix P1invP = getEvects(pcaDims).transpose().mult(projFactorMat);

		// instead of multiplying P1 by invPM 
		// == (reducDataDims X reducDataDims) * (reducDataDims X origDataDims),
		// multiply P1(invP) by M
		// == (reducDataDims X data.numRows()) * (data.numRows() X origDataDims)
		// == fewer calculations
//		Matrix voxSpacePCAEvects = P1invP.mult(dataLoader.getOrigData());
		Matrix voxSpacePCAEvects = P1invP.mult(origData);
		
		eigenvectors = voxSpacePCAEvects.transpose();
		truncateEvalsAndScores(pcaDims);
		pcaInOrigSpace = true;
		
	}
	
	/** Saves PC results to IDL-format files.
	 * @param pcaSavePref prefix (including path) of saved PCA files.  
	 * @param pcaDims - dims to save.  If null, save all.
	 * @param saveAsSplit boolean
	 * @param splitNum
	 * @param splitHalf
	 */
	
	public void savePCAResultsIDL(String pcaSavePref, 
				 int[] pcaDims,
			     boolean saveAsSplit, 
			     int splitNum,
			     int splitHalf) {
//		String pcaEvalFile = "";
		String pcaEvectFile = "";
		String pcaScoreFile = "";
		if (saveAsSplit ) {					
			pcaEvectFile = pcaSavePref + ".PCA." + splitNum + "." + splitHalf
			+ ".evect";
			pcaScoreFile = pcaSavePref + ".PCA." + splitNum + "." + splitHalf 
			+ ".pcScore";

		}
		else {
			pcaEvectFile = pcaSavePref + ".PCA.ALL.evect";
			pcaScoreFile = pcaSavePref + ".PCA.ALL.pcScore";

		}

		if (pcaDims == null) {
			// save all pc dims 
			eigenvectors.printToFile(pcaEvectFile, "IDL");
			pcScores.printToFile(pcaScoreFile, "IDL");
		}
		else {
			Matrix subsetEvects = eigenvectors.subMatrixCols(pcaDims);
			subsetEvects.printToFile(pcaEvectFile, "IDL");
			Matrix subsetPCScores = pcScores.subMatrixCols(pcaDims);
			subsetPCScores.printToFile(pcaScoreFile, "IDL");
		}	
	}
	

	/**
	 * Multiplies values in input double array by input scalar and returns new
	 * double array containing result.
	 * 
	 * @param values -
	 *            Array of values to be scaled by scalar
	 * @param scalar -
	 *            Factor by which to multiply elements in 'values'
	 * @return scaled array
	 */
	private static double[] scale(double[] values, double scalar) {
		double[] scaled_vals = new double[values.length];
		for (int i = 0; i < values.length; ++i) {
			scaled_vals[i] = values[i] * scalar;
		}
		return scaled_vals;
	}
}
