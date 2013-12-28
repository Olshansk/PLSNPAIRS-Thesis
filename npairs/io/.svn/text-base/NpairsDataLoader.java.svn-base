package npairs.io;

import npairs.shared.matlib.*;
import extern.niftijlib.Nifti1Dataset;
import npairs.Npairsj;
import npairs.NpairsjSetupParams;
import npairs.NpairsjException;

import com.jmatio.types.MLDouble;
import extern.NewMatFileReader;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;

import javax.swing.JOptionPane;

import pls.shared.MLFuncs;

/** Loads data volumes listed in input NpairsjSetupParams into 2D Matrix of given type; rows
 *  are volumes and columns are voxels.  Also performs feature selection (e.g.
 *  eigenvalue decomposition) if NpairsjSetupParams includes an initial feature selection step
 *  and stores feature-selection data in 2D Matrix of given type (rows are still volumes; 
 *  columns are feat-selection dimensions).  Also calculates and stores inverse 
 *  feature-selection Matrix.  
*/

public class NpairsDataLoader {
	
	final boolean debug = false;
	
	private Matrix origData;
	
	private Matrix featSelData;
	
	//TODO: Improve clarity/conciseness of this explanation
	/**		
	  		 WHAT IS evdProjFactorMat?
	 
	       Given orig data M = VSUt, let 
	  		featSelData P = MU (i.e P is orig data M in U-space):
	  		PUt = M ==> P1Ut = M1 where P1 is some data in U-space and
				  M1 is P1 data projected back into orig. space
	     	==> need Ut to recover results in orig. space,
	         but Ut too large to store 
	        (U is num orig voxels X reducDataDims <= size(input data)) 
				hence save evdProjFactorMat ( == invP) and
	        reconstruct Ut on the fly;
	        actually, won't bother reconstructing
	        Ut but will calculate P1(invP) instead
	        before multiplying by M, since
	        	Ut = (invP)M 
	        ==> P1Ut = P1(invP)M 
	        ==> by calculating P1(invP) first, only
	            do matrix mult. with huge dim (M.numcols())
	            once.
	        // (NOTE  evdProjFactorMat == ipcmat in IDL NPAIRS).
	        
			If using unweighted ("normed") EVD (i.e. if setupParams.normedEVD == true),
			then S is ignored in projected data representation P, i.e. P = MU(invS) = V.  
	
			This is equivalent to setting all (non-zero diagonal) elements of S to 1, 
			and  gets rid of differences in variance across basis vectors in U; hence 
			this can be thought of as a data "denoising" technique.
			In normed EVD, data is still represented in U-space, hence
			projection back to original space is a projection back through U:
				P1Ut = M1.
			Given P = V = MU(invS), Ut = (invS)(invP)M. 
			Hence we use (invS)(invP) in normed EVD where invP is used
			in regular EVD. But note that normed P = V ==> (invS)(invP) = (invS)Vt,
			whereas in regular EVD, P = VS ==> invP = (invS)Vt! Letting
			P1Ut = P1*Kr*M for some matrix Kr in reg EVD; and P1Ut = P1*Ku*M for
			some matrix Ku in normed EVD, we have that Kr == Ku == (invS)Vt 
			Therefore, we simply store (invS)Vt in evdProjFactorMat in both cases.
	
			 evdProjFactorMat is reducDataDims X data.numRows().
	*/
	
	private Matrix evdProjFactorMat;
	
	private String matlibType;
	private String matlibTypeForInitFeatSel;
	
	private float[] qOffset; 
	private double[] voxelSize;
	private int[] origin; // in 1-relative voxels
	private int[] volDims3D; 
	
	private int[] maskCoords;
	
	private boolean loadDatamats = false;
	
	public NpairsDataLoader(String npairsjSetupParamsMatFilename, 
			String matlibType, String matlibTypeForInitFeatSel, boolean loadDatamats) 
			throws NpairsjException, IOException {
		
		this(new NpairsjSetupParams(npairsjSetupParamsMatFilename, false),
				matlibType, matlibTypeForInitFeatSel, loadDatamats);
	}
	
	public NpairsDataLoader(NpairsjSetupParams nsp, String matlibType,
			String matlibTypeForInitFeatSel, boolean loadDatamats) throws NpairsjException,
		IOException {
			this.matlibType = matlibType;
			this.matlibTypeForInitFeatSel = matlibTypeForInitFeatSel;
			this.loadDatamats = loadDatamats;
			loadNpairsData(nsp);
	}
	
	public Matrix getOrigData() {
		return origData;
	}

	public Matrix getFeatSelData() {
		return featSelData;
	}
	
	public Matrix getEVDProjFactorMat() {
		return evdProjFactorMat;
	}
	
	public boolean loadDatamats() {
		return loadDatamats;
	}
	
	private void loadNpairsData(NpairsjSetupParams setupParams) 
			throws NpairsjException, IOException {
		

		// Load data (each row currently in IDL format: [Y][X][Z]):		
		try {
			double sTime = System.currentTimeMillis();
			double tTime = 0;
			if (!loadDatamats) {
				Npairsj.output.print("Loading data from image files...");
				origData = loadOrigData(setupParams);
				tTime = (System.currentTimeMillis() - sTime) / 1000;
				setDataSpecs(setupParams.getDataFilenames()[0]);
			}
			else {
				Npairsj.output.print("Loading datamats...");
				origData = loadDatamats(setupParams, matlibType);
				tTime = (System.currentTimeMillis() - sTime) / 1000;
//				Npairsj.output.println("Size concatenated dmat data: ");
//				Npairsj.output.println(origData.numRows() + " X " + origData.numCols());

			}
			Npairsj.output.println("[" + tTime + " s]");
			Npairsj.output.println("No. vols: " + origData.numRows());
//			if (debug) {
//				// save it
//				String saveFile = setupParams.resultsFilePrefix + ".origData";
//				System.out.println("Saving orig data to " + saveFile + "...");
//				origData.printToFile(saveFile, "IDL");
//				System.out.println("Finished saving orig data...");
//			}
		}
		catch (OutOfMemoryError e) {
			throw new NpairsjException("Ran out of memory loading data.  Try allocating more " +
					"memory when running plsnpairs.");
		}
		catch (MatrixException e) {
			throw new NpairsjException("Error loading data volumes -  " + e.getMessage());
		}
	
		// Feature Selection:

		Npairsj.output.print("Do initial feature selection? ");
		if (setupParams.initFeatSelect) {
			Npairsj.output.println("Yes");
		}
		else {
			Npairsj.output.println("No");
		}

		
		if (setupParams.initFeatSelect) {
			featSelData = selectFeatures(setupParams);
		}
	}
	
	/** Sets voxel size, origin, qoffset and data dims using given dataset 
	 * 
	 * @param dataFilename
	 */
	private void setDataSpecs(String dataFilename) throws FileNotFoundException, IOException {

		Nifti1Dataset nDS = new Nifti1Dataset(dataFilename);
		nDS.readHeader();
		volDims3D = new int[] {(int)nDS.getXdim(), (int)nDS.getYdim(), (int)nDS.getZdim()};
		float[] voxSizeTmp = nDS.pixdim;
		voxelSize = new double[] {voxSizeTmp[1], voxSizeTmp[2], voxSizeTmp[3]}; // TODO: is this always valid?
		qOffset = nDS.qoffset;
		// take qoffset and calculate origin in voxels
		//  - for each dim, origin = -(qoffset) / (voxsize)
		origin = new int[3];
		for (int i = 0; i < 3; ++i) {
			origin[i] = (int) (-qOffset[i] / voxelSize[i]) + 1; // add one because 1-rel		                                                         // voxels are 1-rel.
		}	
	}
	
	private static Matrix loadDatamats(NpairsjSetupParams setupParams, String matlibType) throws IOException,
		NpairsjException, MatrixException {
		
		String[] datamatFilenames = setupParams.getDatamatFilenames();
		
		// see also ConcatenateDatamat.computeCommonCoordinates()
		int[] andMask = computeCommonCoordinates(datamatFilenames);
		int nRowsAll = 0;
		for (String d : datamatFilenames) {
			NewMatFileReader dmatReader = new NewMatFileReader(d);
			int currNConds = ((MLDouble)dmatReader.getMLArray("st_evt_list")).getN();
			int currWinSz = ((MLDouble)dmatReader.getMLArray("st_win_size")).get(0, 0).intValue();
			nRowsAll += currNConds * currWinSz;
		}
		
		int[] skipTmpts = setupParams.getSkipTmpts(); // which rows of concat. datamat to 
		                                         // exclude
		int nRows = nRowsAll - skipTmpts.length;
		
		Matrix allData = new MatrixImpl(nRows, andMask.length, matlibType).getMatrix();

//		System.out.println("Size of alldata Matrix: " + allData.numRows() + " X " 
//				+ allData.numCols());
		
		int beginRow = 0;
		for (String d : datamatFilenames) {
			NpairsReadDatamat nrd = new NpairsReadDatamat(d, setupParams.conditionSelection);
			Matrix currDatamat = nrd.getDatamat(andMask);
//			System.out.println("Size curr dmat: " + currDatamat.numRows() + 
//					" X " + currDatamat.numCols());
			allData.setSubMatrix(currDatamat, beginRow, 0);
			beginRow += currDatamat.numRows();
		}
		
		return allData;	
	}
	
	// see also ConcatenateDatamat.computeCommonCoordinates()
	private static int[] computeCommonCoordinates(String[] datamatFilenames) throws 
			IOException, NpairsjException {
		int[] dims = ((MLDouble)new NewMatFileReader(datamatFilenames[0]).
				getMLArray("st_dims")).getIntFirstRowOfArray();
		int[] idxCount = new int[dims[0] * dims[1] * dims[2] * dims[3]]; 
		for (String dmfile : datamatFilenames) {
			// check dims
			NewMatFileReader fileReader = new NewMatFileReader(dmfile);
			int[] currDims = ((MLDouble)fileReader.
					getMLArray("st_dims")).getIntFirstRowOfArray();
			if (!Arrays.equals(dims, currDims)) {
				throw new NpairsjException("All input data must be of same resolution.");
			}
			// get coords
			int[] stCoords = ((MLDouble)fileReader.getMLArray("st_coords")).
				getIntFirstRowOfArray();
			for (int c : stCoords) {
				++idxCount[c];
			}
		}
		
		int[] andMask = MLFuncs.find(idxCount, datamatFilenames.length);
		return andMask;
	}
	
	private Matrix loadOrigData(NpairsjSetupParams setupParams) throws MatrixException, IOException {
//		double[][] maskedData = getMaskedData(setupParams);
//		
//		MatrixImpl mImpl = new MatrixImpl(maskedData, matlibType);
//		return mImpl.getMatrix();
		double[] andMask = NiftiIO.getANDMask(setupParams.getMaskFilenames());
		maskCoords = MLFuncs.findNonZero(andMask);				
		Matrix maskedData =  NiftiIO.getMaskedDataMat(setupParams, matlibType, maskCoords);
	
		return maskedData;
	}
	
	
	/** Returns Matrix containing feature-selected data using feature selection 
	 * technique selected in setupParams.  
	 * REQUIRED: 0.0 < nsp.dataReductionFactor <= 1.0
	 */
	private Matrix selectFeatures(NpairsjSetupParams setupParams) 
		throws IOException, NpairsjException {
	
		Matrix featSelData = null;
		
		if (setupParams.initEVD) {
			if (setupParams.loadEVD) {
//				if (debug) {
				Npairsj.output.print("Loading EVD from files... ");
//				}
				double sTime = System.currentTimeMillis();
				
				String svdEvectsFilename = setupParams.evdFilePref + ".EVD.evects";
				Matrix svdEvects = null;
				double[] svdEvals1D = null;
				double[][] svdEvects2D = NpairsjIO.readFromIDLFile(svdEvectsFilename);
				svdEvects = new MatrixImpl(svdEvects2D).getMatrix();

				String svdEvalsFilename = setupParams.evdFilePref + ".EVD.evals";
				double[][] svdEvals2D = NpairsjIO.readFromIDLFile(svdEvalsFilename);
				svdEvals1D = MatlabMatrix.trimTo1D(svdEvals2D);
				
//				Matrix evalMat = new MatrixImpl(svdEvals1D.length, svdEvals1D.length).getMatrix();
//				evalMat.diag(svdEvals1D);
				
				double reducFactor = setupParams.dataReductionFactor;		
				int reducDataDims = 0;
				
				if (origData.numRows() < origData.numCols()) {
					reducDataDims = (int)Math.round(origData.numRows() * reducFactor);
				}
				else {
					reducDataDims = (int)Math.round(origData.numCols() * reducFactor);
//					throw new NpairsjException("WARNING: input data has more " +
//							"scans than masked voxels!");
				}
				
				int[] rowRange = new int[] {0, origData.numRows() - 1};
				int[] reducColRange = new int[] {0, reducDataDims - 1};
				Matrix reducDimEvects = svdEvects.subMatrix(rowRange, reducColRange);
				
				// get singular values by finding sqrts of svd evals
				Matrix S = new MatrixImpl(reducDataDims, reducDataDims).getMatrix();
				for (int i = 0; i < reducDataDims; ++i) {
					S.setQuick(i,i, Math.sqrt(svdEvals1D[i]));
				}
				if (setupParams.normedEVD) { 
					featSelData = reducDimEvects;
				}
				else {
					featSelData = reducDimEvects.mult(S);	
				}

				Matrix invS = new MatrixImpl(reducDataDims, reducDataDims).getMatrix();
				for (int i = 0; i < reducDataDims; ++i) {
					invS.setQuick(i, i, 1 / S.getQuick(i, i));
				}

				evdProjFactorMat = invS.mult(reducDimEvects.transpose());
				double tTime = (System.currentTimeMillis() - sTime) / 1000;
				
				Npairsj.output.println("[" + tTime + " s]");
				
//					int hr = (int)(tTime / 3600);
//					int min = (int)(tTime / 60) - (hr * 60);
//					double s = tTime - (hr * 3600) - (min * 60) ;
//					System.out.print("Total time of EVD: " + hr + " h " +
//							min + " min ");
//					System.out.printf("%.3f", s);
//					System.out.println(" sec");
				
			}
			else {
				featSelData = selectFeaturesEVD(setupParams, origData, 
						setupParams.dataReductionFactor);
			}
		}
		
		return featSelData;
	}

	
	/** Returns Matrix containing feature-selected data using eigenvalue decomposition.
	 *  Note that IDL uses EVD, too (actually eigenql of non-mc ssp mat) although it's
	 *  called 'SVD' in IDL NPAIRS. 
	 *   
	 * REQUIRED: 0.0 < dataReductionFactor <= 1.0
	 */
	private Matrix selectFeaturesEVD(NpairsjSetupParams setupParams, Matrix data, 
			double dataReductionFactor) throws IOException, NpairsjException {
		
		String taskMsg = "Running initial EVD with DRF = " + 
		dataReductionFactor + "... ";
		if (setupParams.normedEVD) {
			taskMsg = "Running initial EVD (normed) with DRF = " + 
			dataReductionFactor + "... ";
		}
		Npairsj.output.println(taskMsg);
//			System.out.println("Using " + matlibTypeForInitFeatSel + " Matrix library " +
//					"to do EVD...");
		
		Matrix featSelData = null;
		double reducFactor = dataReductionFactor;		
		
		int reducDataDims = (int)Math.round(Math.min(data.numRows(), data.numCols()) * reducFactor);
			
/*			 Given data = M, PCA(Mt) ==> MMt = V(S^2)Vt, where S^2 == diag Matrix containing
			 squared singular values from Matrix S in SVD(M) = VSUt along diagonal.
			 ==> featSelData = proj. P of data M onto eigenimages U, 
			 i.e., featSelData P = MU
			                     = VSUtU 
			                     = VS, 
			 where U is data.numCols() X reducDataDims(rDD);
			      
			 		 S is rDD X rDD;
			 		 
			       V is data.numRows() X rDD;
			 			
			 ==> P = VS is data.numRows() X rDD
			       == dim(MU) as required */
			
			double sTime = System.currentTimeMillis();
			EigenvalueDecomposition evd = null;
			Matrix svdEvals = null;
			Matrix svdEvects = null;
			Matrix invSqrtEvals = null;
			if (!matlibType.equalsIgnoreCase(matlibTypeForInitFeatSel)) {
				//TODO: too many copies of data being made here!
				if (matlibTypeForInitFeatSel.equalsIgnoreCase("MATLAB")) {
					if (debug) {
						System.out.println("Doing EVD in Matlab...");
					}
					MatlabMatrix mlData = new MatlabMatrix(data.toArray());
					Npairsj.output.print("\tCreating SSP matrix from data matrix...");
					double sspSTime = System.currentTimeMillis();
					MatlabMatrix sspMat = mlData.mult(mlData.transpose());
					double sspTTime = (System.currentTimeMillis() - sspSTime) / 1000;
					Npairsj.output.println("[" + sspTTime + "s]");
					Npairsj.output.print("\tRunning EVD on SSP matrix...");
					double evdSTime = System.currentTimeMillis();
					evd = sspMat.eigenvalueDecomposition();
					double evdTTime = (System.currentTimeMillis() - evdSTime) / 1000;
					Npairsj.output.println("[" + evdTTime + "s]");
				}
				if (matlibTypeForInitFeatSel.equalsIgnoreCase("COLT")) {
					if (debug) {
						System.out.println("Doing EVD in Colt...");
					}
					ColtMatrix coltData = new ColtMatrix(data.toArray());
					Npairsj.output.print("\tCreating SSP matrix from data matrix...");
					double sspSTime = System.currentTimeMillis();
					ColtMatrix sspMat = coltData.mult(coltData.transpose());
					double sspTTime = (System.currentTimeMillis() - sspSTime) / 1000;
					Npairsj.output.println("[" + sspTTime + "s]");
					Npairsj.output.print("\tRunning EVD on SSP matrix...");
					double evdSTime = System.currentTimeMillis();
					evd = sspMat.eigenvalueDecomposition();
					double evdTTime = (System.currentTimeMillis() - evdSTime) / 1000;
					Npairsj.output.println("[" + evdTTime + "s]");
				}
				svdEvals = new MatrixImpl(evd.getRealEvalMat().toArray()).getMatrix();
				svdEvects = new MatrixImpl(evd.getEvects().toArray()).getMatrix();
				invSqrtEvals = new MatrixImpl(evd.getInvSqrtRealEvalMat().toArray()).getMatrix();
			}
			else {
				Npairsj.output.print("\tCreating SSP matrix from data matrix...");
				double sspSTime = System.currentTimeMillis();
				Matrix sspData = data.sspByRow();
				double sspTTime = (System.currentTimeMillis() - sspSTime) / 1000;
				Npairsj.output.println("[" + sspTTime + "s]");
				Npairsj.output.print("\tRunning EVD on SSP matrix...");
				double evdSTime = System.currentTimeMillis();
				evd = sspData.eigenvalueDecomposition();
				double evdTTime = (System.currentTimeMillis() - evdSTime) / 1000;
				Npairsj.output.println("[" + evdTTime + "s]");
				svdEvals = evd.getRealEvalMat();
				svdEvects = evd.getEvects();
				invSqrtEvals = evd.getInvSqrtRealEvalMat();
			}

//			if (debug) {
				double tTime = (System.currentTimeMillis() - sTime) / 1000;
				int hr = (int)(tTime / 3600);
				int min = (int)(tTime / 60) - (hr * 60);
				double s = tTime - (hr * 3600) - (min * 60) ;
				Npairsj.output.print("Total time doing EVD calculations: [" + hr + " h " +
						min + " min ");
				Npairsj.output.printf("%.3f", s);
				Npairsj.output.println(" s]");
//			}
			
			// save evals/evects, i.e. S^2 and V (TODO: save only if user specifies to do so?):
			String evalsFilename = setupParams.resultsFilePrefix + ".EVD.evals";
			String evectsFilename = setupParams.resultsFilePrefix + ".EVD.evects";
			double saveSTime = System.currentTimeMillis();
			Npairsj.output.print("Saving EVD info to file...");
			NpairsjIO.printToIDLFile(evd.getRealEvals(), evalsFilename);
			svdEvects.printToFile(evectsFilename, "IDL");
			double saveTTime = (System.currentTimeMillis() - saveSTime) / 1000;
			Npairsj.output.println("[" + saveTTime + " s]");
			
			//***********************************************************
			// Save eigims too!
			//***********************************************************
//			System.out.print("Creating and saving svd eigims...");
//			double strtTime = System.currentTimeMillis();
//			String eigimsFilename = setupParams.resultsFilePrefix + ".EVD.eigims";
//			Matrix svdEigims = data.transpose().mult(svdEvects).mult(invSqrtEvals);
//			svdEigims.printToFile(eigimsFilename, "IDL");
//			double totTime = (System.currentTimeMillis() - strtTime) / 1000;
//			System.out.println("[" + totTime + " s]");
			int[] rowRange = new int[] {0, data.numRows() - 1};
			int[] reducColRange = new int[] {0, reducDataDims - 1};
			
			if (setupParams.normedEVD) {
				// ignore S so featSelData = V
				featSelData = svdEvects.subMatrix(rowRange, reducColRange);
			}
			else {
				// Given data M = VSUt, consider basis space to be
				// U, hence MU = VS, i.e., featSelData = VS
				Matrix S = svdEvals.subMatrix(reducColRange, reducColRange);
				for (int i = 0; i < reducDataDims; ++i) {
					double currEval = S.get(i, i);
					if (currEval > 0) {
						S.set(i, i, Math.sqrt(S.get(i, i)));
					}
					else S.set(i, i, 0);
				}

				featSelData = svdEvects.subMatrix(rowRange, reducColRange).
					mult(S);
			}
			
			double iSTime = System.currentTimeMillis();
			Npairsj.output.print("Calculating inverse EVD matrix ...");
			evdProjFactorMat = invSqrtEvals.subMatrix(reducColRange, reducColRange).
					mult(svdEvects.subMatrix(rowRange, reducColRange).transpose());
			double iTTime = (System.currentTimeMillis() - iSTime) / 1000;
			Npairsj.output.println("[" + iTTime + "s]");
			        					
//		}
		
		return featSelData;
	}
	
	public float[] getQOffset() {
		return qOffset;
	}
	public double[] getVoxSize() {
		return voxelSize;
	}
	
	public int[] getDims() {
		return volDims3D;
	}
	
	public int[] getOrigin() {
		return origin;
	}
	
	public int[] getMaskCoords() {
		return maskCoords;
	}
	
	public static void main(String[] args) {
		String setupParamsFile = args[0];
		try {
			Matrix datamat = loadDatamats(new NpairsjSetupParams(setupParamsFile, true), "Colt");
			System.out.println("Size of concatenated datamat: " + datamat.numRows() +
				" X " + datamat.numCols());
			datamat.printToFile("/home/anita/Desktop/concat_dmat.2D", "IDL");
		}
		catch (Exception e) {
			e.printStackTrace();
		}
	}
}
