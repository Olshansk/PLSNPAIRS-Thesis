package npairs;

import java.io.Serializable;

import npairs.shared.matlib.*;
import npairs.utils.*;


// import npairs.io.*;
import pls.shared.MLFuncs;
public class Analysis implements Serializable{

	private boolean debug = true; 
	
	private NpairsjSetupParams setupParams;

	private Matrix data;  // Data to be analyzed.  If PCA is run, then data is set to
	                      // PC scores (for PCDimsForCVA dims, if CVA run; for all input
	                      // col dims, otherwise.)
	
	private int[] PCDimsForCVA;  // 0-relative
	
	private int[] cvaClassLabels;  
	
	private int[] sessionLabels; // needed for mean session scan removal
	
	private boolean useQuadCVA; //= true; // Added by Grigori on April 20, 2009
	
	private CVA currCVA = null; // holds cva results if CVA is run 
							    // except for eigenimages; see cvEigims)
	
	private PCA currPCA = null; // holds pca results if PCA is run
	
	private Analysis refAnalysis = null; // holds reference Analysis if results from
	                                     // this Analysis must be matched to it
	                                     // (e.g. if CV dims of current CVA analysis
	                                     // are to be negated/permuted to match
	                                     // CV dims in reference CVA analysis)
	
	private static int splitAnalysisNum = 0;
	
	//private static int count = 0; // FOR DEBUGGING
	
//	private NpairsDataLoader dataLoader;
	
//	private boolean isFirst = false;
	
	/***************************************************************************************
	 * Class Analysis() analyzes input data using model(s) described in input 
	 * NpairsjSetupParams
	 * - includes methods to generate stats (Prediction, e.g.)
	 * - includes feature selection method
	 **************************************************************************************/

	/********************
	 *  Two constructors:
	 ********************/

	//	 (1) To do full-data analysis: 
//	public Analysis (Matrix data, NpairsjSetupParams setupParams, NpairsDataLoader dataLoader) {
//		this.setupParams = setupParams;
//		this.data = data;
//		this.PCDimsForCVA = setupParams.cvaPCSetAll;
//		this.cvaClassLabels = setupParams.cvaClassLabels;
//		this.sessionLabels = setupParams.sessionLabels;
//		this.dataLoader = dataLoader;
//	}
	
	// (1) To do full-data analysis: 
	public Analysis (Matrix data, NpairsjSetupParams setupParams) {
		this.setupParams = setupParams;
		this.data = data;
		this.PCDimsForCVA = setupParams.cvaPCSetAll;
		this.cvaClassLabels = setupParams.getClassLabels();
		this.sessionLabels = setupParams.getSessLabels();
		this.useQuadCVA = setupParams.useQuadCVA;
	}

	// (2) To do split-data analysis:
	public Analysis (Matrix data, NpairsjSetupParams setupParams, 
			int[] currSplitIndices, boolean isFirst, Analysis refAnalysis) {
		++splitAnalysisNum;
		this.setupParams = setupParams;
		this.data = data.subMatrixRows(currSplitIndices);
		this.cvaClassLabels = MLFuncs.getItemsAtIndices(setupParams.getClassLabels(), currSplitIndices);
		this.sessionLabels = MLFuncs.getItemsAtIndices(setupParams.getSessLabels(), currSplitIndices);
		if (isFirst) {
			this.PCDimsForCVA = setupParams.cvaPCSet1;
		}
		else {
			this.PCDimsForCVA = setupParams.cvaPCSet2;
		}
		
		this.refAnalysis = refAnalysis;
		this.useQuadCVA = setupParams.useQuadCVA;
		
//		this.isFirst = isFirst;
		
	}
	

	/**************************************************************************************
	 * methods:
	 *****************************/

	public void run() throws NpairsjException {
		
		Matrix pcaEvects = null; // required in case results need to be transformed 
						         // back to orig space from pca space, i.e. if cva 
		                         // follows pca
		
		if (setupParams.preProcess) {
			preprocessData();      
		}

//		if (setupParams.glmRun) {
//		// initialize glm params and then do GLM analysis
//       (not implemented yet) 		
//		}

		if (setupParams.pcaRun) {

			// DEBUGGING
//			count = count + 1;
//			String filename = "features"+count+".txt";
//			data.printToFile(filename, "IDL");
			// END OF DEBUGGING
			
			
			double sTime = System.currentTimeMillis();
			Npairsj.output.println("Running PCA... ");
			currPCA = new PCA(data, setupParams.normalizePCsBySD, !setupParams.initFeatSelect);
			double tTime = (System.currentTimeMillis() - sTime) / 1000;
			Npairsj.output.println("Total time PCA: [" + tTime + " s]");
			
			pcaEvects = currPCA.getEvects(PCDimsForCVA);
			if (setupParams.cvaRun) {
				data = currPCA.getPCScores(PCDimsForCVA);
			}
			else {
				data = currPCA.getPCScores();
			}
			
			// DEBUGGING
//			System.out.println ("Norma " + setupParams.normalizePCsBySD);
//			filename = "PCscores"+count+".txt";
//			data.printToFile(filename, "IDL");
			// END OF DEBUGGING			
			
			
		}

		if (setupParams.cvaRun) {
			// If pca has been run first, then data will be in 
			// pca space.
			double sTime = System.currentTimeMillis();
			Npairsj.output.print("Running CVA... ");
			
			/// added by Grigori, April 20, 2009
			if (useQuadCVA)				
				//currCVA = new JavaQuadCVA(data, cvaClassLabels);
				currCVA = new MatlabQD (data, cvaClassLabels, pcaEvects);
			else
				currCVA = new CVA(data, cvaClassLabels, setupParams.saveLotsOfFiles, !setupParams.initFeatSelect); 
			
			double tTime = (System.currentTimeMillis() - sTime) / 1000;
			Npairsj.output.println("Total time CVA: [" + tTime + " s]");
			
			if (setupParams.pcaRun) {
				// rotate eigenimages back from pca space
				sTime = System.currentTimeMillis();
				if (debug) {
					Npairsj.output.print("Creating CVA eigenimages... ");
				}
				boolean pcaEvectsInOrigSpace = !setupParams.initFeatSelect;
				currCVA.createFeatSpEigenimages(pcaEvects, pcaEvectsInOrigSpace);
				if (debug) {
					tTime = (System.currentTimeMillis() - sTime) / 1000;
					Npairsj.output.println("["+ tTime + " s]");
				}
			}
			else {
				// cva eigenimages == cva eigenvectors
				currCVA.createFeatSpEigenimages(null, false);
			}		
		}
		// can add more techniques here later
		
		if (refAnalysis != null) {
			
			double sTime = System.currentTimeMillis();
			if (debug) {
				Npairsj.output.println("Matching split analysis to reference... ");
			}
			boolean useEigims = setupParams.useEigimsInProcrust();
			boolean useFullProc = false;
			this.matchToRef(useEigims, useFullProc);
			if (debug) {
				double tTime = (System.currentTimeMillis() - sTime) / 1000;
				Npairsj.output.println("[" + tTime + " s] (Match to ref)");			
			}
		}

	} // end method run()

	/** Match results in this Analysis to input refAnalysis
	 using Procrustes()
	 If useEigims, useFullProcrustes == false:
	 	Uses technique used in IDL npairs cva_reference.pro code:
	 apply Proc. to mean cv scores (ref. + curr.) to determine
	 whether each CV dim is to be negated or permuted; apply
	 neg. and perm. to curr. evals, evects (eigenimages), cv scores
	 and chi-squared data for each CV dim.
	 Else if useEigims == true, apply Proc. to cv eigenimages 
	 (ref. and curr.); if useFullProcrustes == true, use Xform
	 instead of just doing negation and permutation.
	 *
	 * @param useEigims if true, use eigims instead of average cv scores to calculate Procrustes transformation
	 * @param useFullProcrustes if true, apply Procrustes transformation instead of just negating
	 * 					and permuting 
	 * 
	*/
	
	protected void matchToRef(boolean useEigims, boolean useFullProcrustes) throws NpairsjException {

		// Added by Grigori
		if (useQuadCVA) {
			return; // no Procrustes for QD
		}
			
		
		// CVA
		if (setupParams.cvaRun) {
			CVA refCVA = refAnalysis.getCVA();
			//System.out.println("%%%%%REF: "+refCVA.getNumCVDims()+"CVA: "+currCVA.getNumCVDims());
			if (refCVA.getNumCVDims() < currCVA.getNumCVDims()) {
				throw new NpairsjException("Error - reference CVA must have at least " +
						"as many dimensions as current CVA");
			}
			
			double sTime = System.currentTimeMillis();
			Procrustes proc;
			if (!useEigims) {
				// Get average CV scores for each input condition
				Matrix avgRefCVScores = refCVA.avgCVScores();
				Matrix currAvgCVScores = currCVA.avgCVScores();
				avgRefCVScores = avgRefCVScores.subMatrix(new int[] {0, avgRefCVScores.numRows() - 1},
						new int[] {0, currAvgCVScores.numCols() - 1});

				// apply Procrustes to the mean CV scores
				Npairsj.output.print("Calculating Procrustes using average CV scores... ");
				sTime = System.currentTimeMillis();
				proc = new Procrustes(avgRefCVScores.transpose(), 
						currAvgCVScores.transpose());
				double tTime = (System.currentTimeMillis() - sTime) / 1000;				
				Npairsj.output.println("[" + tTime + " s]");
			}
			else {
				// get CV eigenimages (in feat space if performed)
				Matrix refCVEigims = refCVA.getFeatSpEigims();
				Matrix currCVEigims = currCVA.getFeatSpEigims();
				// apply Procrustes to eigenimages
				Npairsj.output.print("Calculating Procrustes using CV eigenimages.. ");
				sTime = System.currentTimeMillis();
				proc = new Procrustes(refCVEigims.transpose(), currCVEigims.transpose());
				double tTime = (System.currentTimeMillis() - sTime) / 1000;	
				Npairsj.output.println("[" + tTime + " s]");			
			}
			
			if (!useFullProcrustes) {
				int[] sign = proc.getRefSign();
				int[] permut = proc.getRefPermute();

	//			if (debug) {
					Npairsj.output.print("Signs: ");
					npairs.io.NpairsjIO.print(sign, Npairsj.output);
					Npairsj.output.print("Permutations: ");
					npairs.io.NpairsjIO.print(permut, Npairsj.output);
	//			}
				
				Npairsj.output.print("Negating and permuting dims... ");
				sTime = System.currentTimeMillis();
				currCVA.negateAndPermuteDims(sign, permut);

				double ttTime = (System.currentTimeMillis() - sTime) / 1000;
				Npairsj.output.println("[" + ttTime + " s]");
			}
			else {
				// not implemented! 
				currCVA.applyFullProcrust(proc);
			}
		}
	}
		
	private void preprocessData() throws NpairsjException {
		// do MSR (mean session scan removal)? 
		if (setupParams.removeSessionMeans) {
			data = removeMeanSessionScans(sessionLabels, data);
		}
		// more to be implemented
	}
	
	/** Remove mean scan from each session to get rid of session effects. Algorithm used is same
	 * as what is used in idl NPAIRS ssm_xform.pro (see 'transf' option 8: 'subtract out subject mean 
	 * profiles') 
	 */
	protected Matrix removeMeanSessionScans(int[] sessionLabels, Matrix data) 
			throws NpairsjException {
		int[] uniqSessLabs = MLFuncs.unique(sessionLabels);
		int nSess = uniqSessLabs.length;
		int nDataDims = data.numCols();
		
		if (nSess > 1) {
			double[] grandMean = new double[nDataDims];
			// mean session scans are held in columns of sessMeans, not rows
			Matrix sessMeans = new MatrixImpl(nDataDims, nSess).getMatrix();
			int[] grandCount = new int[nDataDims];
		
			for (int s = 0; s < nSess; ++s) {
				int[] tmpCount = new int[nDataDims];
				int[] currScanLocs = MLFuncs.find(sessionLabels, uniqSessLabs[s]);
				int nCurrScans = currScanLocs.length;
				for (int i = 0; i < nCurrScans; ++i) {
					// add current scan data to running total for curr session in sessMeans
					double[] currScanData = data.getRow(currScanLocs[i]);
					int[] whereNZ = MLFuncs.findNonZero(currScanData);
				
					double[] currSessSumScans = sessMeans.getColumn(s);
					int nNZ = whereNZ.length;
					for (int n = 0; n < nNZ; ++n) {
						currSessSumScans[whereNZ[n]] += currScanData[whereNZ[n]];
						// increment element counter where curr scan has non-zero values	
						tmpCount[whereNZ[n]] += 1;
					}
					// set sessMeans to cumulative scan total for this session (for now)
					sessMeans.setColumn(s, currSessSumScans);
								
				}
				int[] currSessNZ = MLFuncs.findNonZero(tmpCount);
				if (currSessNZ.length > 0) {
					
					double[] currSessScanSum = sessMeans.getColumn(s);
					double[] currSessAvgScan = new double[nDataDims];
					for (int nzLoc : currSessNZ) {
						currSessAvgScan[nzLoc] = currSessScanSum[nzLoc] / tmpCount[nzLoc];	
						grandMean[nzLoc] += currSessScanSum[nzLoc];
						grandCount[nzLoc] += tmpCount[nzLoc];
					}
					sessMeans.setColumn(s, currSessAvgScan);		
				}
			}
			// grandMean == avg scan (actually avg of all non-zero values at each voxel 
			// or equivalent data basis component)
			int[] whereNZ = MLFuncs.findNonZero(grandCount);
			for (int nzLoc : whereNZ) {
				grandMean[nzLoc] /= grandCount[nzLoc];
			}
			
			// Remove session means from data (leaving in grand mean).  This
			// will not change the grand mean; it just means that each session 
			// mean will equal the grand mean.
			for (int s = 0; s < nSess; ++s) {
				// diffs is grandmean - sessmean, so e.g. if grandmean is 0, then 
				// diffs is -sessmean, hence *adding* diffs to curr scan data results
				// in removing current sessmean 
				double[] diffs = MLFuncs.subtract(grandMean, sessMeans.getColumn(s));
				int[] currScanLocs = MLFuncs.find(sessionLabels, uniqSessLabs[s]);
				int nCurrScans = currScanLocs.length;
				for (int i = 0; i < nCurrScans; ++i) {
					double[] currScanData = data.getRow(currScanLocs[i]);
					for (int j = 0; j < nDataDims; ++j) {
						if (currScanData[j] != 0) {
							currScanData[j] += diffs[j];
						}
					}
					data.setRow(currScanLocs[i], currScanData);
				}
			}
			
		} // end else
		
		// print loc of zeros for each data dim
//		System.out.println("Num scans: " + data.numRows());
//		for (int j = 0; j < nDataDims; ++j) {
//			int[] whereNZ = MLFuncs.findNonZero(data.getColumn(j));
//			int numZ = data.numRows() - whereNZ.length;
//			System.out.println("No. zeros (dim " + j + "): " + numZ);
//		}
		
//		// fill in zeros by regressing global mean against regional ('voxel') values
//		int nScans = data.numRows();
//		double[] scanMeans = new double[data.numRows()];
//		for (int i = 0; i < nScans; ++i) {
//			scanMeans[i] = MLFuncs.avg(data.getRow(i));
//		}
//		for (int j = 0; j < nDataDims; ++j) {
//			int[] whereNZ = MLFuncs.findNonZero(data.getColumn(j));
//			if (whereNZ.length > 0 && whereNZ.length < nScans) {	
//				// some but not all scans have 0 in current dim;
//				// extract the non-zero scan elements in curr dim and the
//				// corresponding scan means
//				double[] meansNZ = MLFuncs.getItemsAtIndices(scanMeans, whereNZ);
//				double[] dataNZ = data.subMatrix(whereNZ, new int[] {j}).getColumn(0);
//			}
//		}
		
		// now remove grand mean (i.e., column-centre the data).  * ASSUMES NO ZEROS *
		// TODO: consider whether there ever could be zeros in masked data; if not, don't
		// need non-zero-finding techniques above, either.
		// (NOTE: idl ssm_xform fills in zeros by regressing global mean against regional
		// ('voxel') values)
		data = data.meanCentreColumns();
		
		return data;
	}
	
	
	/** Returns null if no pca has been run in current Analysis
	 * 
	 * @return PCA if run; null otherwise
	 */
	public PCA getPCA() {
		return currPCA;
	}
	
	/** Returns null if no cva has been run in current Analysis
	 * 
	 * @return CVA if run; null otherwise
	 */
	public CVA getCVA() {
		return currCVA;
	}	
	
//	/** Returns input data used for this Analysis 
//	 *  (In split-data Analysis case, returns subset of input
//	 *   data entered into current Analysis)
//	 */
//	public Matrix getInputData() {
//		return data;
//	}
	
//	/** Returns average CV Scores for each condition (class label)
//	 * @param Matrix cvScores - dims [num scans][num cv dims]
//	 * @return Matrix containing avg cv scores for each condition (class label)
//	 *                    - dims [num conds][num cv dims]
//	 * @see ResultSaver.avgCVScores(double[][], int[], int[], int[])
//	 */
//	private Matrix avgCVScores(Matrix cvScores) {
//			
//		return avgCVScores(cvScores, cvaClassLabels);
//		Hashtable<Integer, int[]> condIndices = CVA.getLabelIndices(cvaClassLabels);
//		int numCond = condIndices.size();
//		int numCVDims = cvScores.numCols();
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
//				double[] currScores = MLFuncs.getItemsAtIndices(cvScores.getColumn(dim),
//						currCondIndices);
//				avgScores.set(cond, dim, MLFuncs.avg(currScores));
//			}
//		}
//		
//		return avgScores;
//	}
	
//	/** Returns average CV Scores for each condition (class label)
//	 * @param Matrix cvScores - dims [num scans][num cv dims]
//	 * @return Matrix containing avg cv scores for each condition (class label)
//	 *                    - dims [num conds][num cv dims]
//	 * @see ResultSaver.avgCVScores(double[][], int[], int[], int[])
//	 */
//	private static Matrix avgCVScores(Matrix cvScores, int[] classLabels) {
//			
//		Hashtable<Integer, int[]> condIndices = CVA.getLabelIndices(classLabels);
//		int numCond = condIndices.size();
//		int numCVDims = cvScores.numCols();
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
//				double[] currScores = MLFuncs.getItemsAtIndices(cvScores.getColumn(dim),
//						currCondIndices);
//				avgScores.set(cond, dim, MLFuncs.avg(currScores));
//			}
//		}
//		
//		return avgScores;
//	}
	
	
	
//	// for testing
//	public static void main(String[] args) {
//		
//		// test avg cv scores
//		try {
//			int[] clsLabs = npairs.io.NpairsjIO.readIntsFromFile("\\anita\\workspace\\" +
//				"PLSNPAIRSGoogleRepo\\localTestData\\classFile_9cond_89scan.txt");
//			int[] inclLabs = new int[189];
//			int count = 0;
//			for (int i = 0; i < clsLabs.length; ++i) {
//				if (clsLabs[i] > 0) {
//					inclLabs[count] = clsLabs[i];
//					inclLabs[count + 63] = clsLabs[i];
//					inclLabs[count + 126] = clsLabs[i];
//					++count;
//				}
//			}
//			Npairsj.output.println("Labels: ");
//			npairs.io.NpairsjIO.print(inclLabs);
//			
//			String cvFile = "\\anita\\workspace\\PLSNPAIRSGoogleRepo\\localTestData\\" +
//				"localTestResults\\y246_mar1_9class\\y246_mar1_9class_NPAIRSJresult_" +
//				"NPAIRSJresult.CVA.ALL.can";
//			double[][] cvs = npairs.io.NpairsjIO.readFromIDLFile(cvFile);
//			Matrix cvmat = new MatrixImpl(cvs, "COLT").getMatrix();
//			Npairsj.output.println("Size cv mat: " + cvmat.numRows() + " X " + cvmat.numCols());
//			Matrix avg = avgCVScores(cvmat, inclLabs);
//			Npairsj.output.println("Avg cv scores: ");
//			avg.print();
//			avg.printToFile("\\anita\\workspace\\PLSNPAIRSGoogleRepo\\localTestData\\" +
//					"testAvgCVs.2D", "IDL");
//		}
//		catch (MatrixException m) {
//			m.printStackTrace();
//		}
//		catch (java.io.IOException e) {
//			e.printStackTrace();
//		}
//		catch (NpairsjException e) {
//			e.printStackTrace();
//		}

	
}



