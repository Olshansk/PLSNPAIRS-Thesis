package npairs.utils;

import pls.shared.MLFuncs;
import npairs.shared.matlib.Matrix;
import npairs.shared.matlib.MatrixImpl;
import npairs.NpairsjException;

import java.util.Arrays;

// Generates prediction statistics for input NPAIRS training and test data.
// If test vol is equally close to multiple training mean class CV scores, test vol is
// assigned to first class found satisfying minimum proximity measure

public class PredictionStats {
	
	boolean debug = false;
	
	double[] priors;  	// - contains prior probabilities for each cva class
	                    // ( ~ proportion of total data belonging to each class)
	                    // - priors are given in ascending order of class labels
	
	double[][] bwStats; // dims of bwStats: (no. CV dims) rows X 3
	                    // bwStats(k, 0) = between-class variation metric
	                    // bwStats(k, 1) = within-class variation metric
	                    // bwStats(k, 2) = ratio b/w
	                    // bsStats(k, 3) = percent misclassified
	                    // See IDL code npairs_cva_general.pro for details
	
	// 1st dim size in following 5 arrays is 2: 1st == with priors; 2nd == without priors
	double[][] ppTrueClass;    // posterior probs of each test scan belonging to true class
	
	double[][] sqrdPredError;  // (1-ppTrueClass)^2 for each test vol
	
	double[][][] ppAllClasses; // no. classes X no. test vols
	
	double[][] predClass;      // predicted class label for each test vol
	
	double[][] correctPred;    // for each test vol, 1 if correct; 0 if not
	
	//private static int count = 0; // FOR DEBUGGING

	
	
	public PredictionStats(Matrix cvsTrain, Matrix cvsTest, 
			               int[] trainVols, int[] testVols,
			               int[] cvaClassLabels) throws NpairsjException {
		
		// make sure training and test have same no. of dims
		int nCVDims = cvsTrain.numCols();
		if (nCVDims != cvsTest.numCols()) {
			throw new IllegalArgumentException("Training and test data must have same column (CV) " +
					"dimensions");
		}
		
		// trim cvs of zero padding (only include scans in current train or test set)
		cvsTrain = cvsTrain.subMatrixRows(trainVols);
		cvsTest = cvsTest.subMatrixRows(testVols);
		
		// get training and test class labels
		int[] classLabTr = MLFuncs.getItemsAtIndices(cvaClassLabels, trainVols);
		int[] classLabTest = MLFuncs.getItemsAtIndices(cvaClassLabels, testVols);
		
		int[] uniqClsLabTr = MLFuncs.sortAscending(MLFuncs.unique(classLabTr));
		int[] uniqClsLabTest = MLFuncs.sortAscending(MLFuncs.unique(classLabTest));
		int nClassIncl = uniqClsLabTr.length;
		int nVolsTest = testVols.length;
		if (!Arrays.equals(uniqClsLabTr, uniqClsLabTest)) {
			// TODO: consider - could have more labels in training set?
			throw new IllegalArgumentException("Training and test volumes must have same class labels");
		}
		
        // calculate priors
		int[] uniqClassLab = MLFuncs.sortAscending(MLFuncs.unique(cvaClassLabels));
		int nClasses = uniqClassLab.length;
		double nVols = (double)cvaClassLabels.length;
		
		priors = new double[nClassIncl];
		for (int c = 0; c < nClassIncl; ++c) {
			int[] currClassLoc = MLFuncs.find(cvaClassLabels, uniqClsLabTr[c]);
			double szCurrClass = (double) currClassLoc.length;
			priors[c] = szCurrClass / nVols;
			
			// DEBUGGING
			//System.out.println (szCurrClass + ", " + nVols);
			// END OF DEBUGGING
		}
		
		if (debug) {
			System.out.print("Priors: ");
			npairs.io.NpairsjIO.print(priors);
		}
		
		// initialize
		bwStats = new double[nCVDims][4];	
		ppTrueClass = new double[2][nVolsTest];
		sqrdPredError = new double[2][nVolsTest];
		ppAllClasses = new double[2][nVolsTest][nClasses]; // for excluded classes, set to 0 
		predClass = new double[2][nVolsTest];
		correctPred = new double[2][nVolsTest];
		
		// compute class mean CV scores for training data
		double sTime = 0;
		if (debug) {
			sTime = System.currentTimeMillis();
			System.out.print("Getting mean CVs... ");
		}
		Matrix clsMeanCVsTrain = getMeanCVs(cvsTrain, classLabTr); // dims = nClassesIncl X nCVDims
		if (debug) {
			double tTime = (System.currentTimeMillis() - sTime) / 1000;
			System.out.println("[" + tTime + "]");
			System.out.println("Mean CVs (training): ");
			clsMeanCVsTrain.print();
		}
		   
		// compute between and within class variation for test CVs
		double[] cvGrandMeansTest = cvsTest.colMeans();
		
		for (int d = 0; d < nCVDims; ++d) {
			if (debug) {
				System.out.print("Computing between and within class variation for test CVs dim " + d);
				sTime = System.currentTimeMillis();
			}
			double b = 0.0;
			double w = 0.0;
			
			Matrix currDimCVsTest = cvsTest.subMatrixCols(new int[] {d}); // note currDimCVsTest has
			                                                              // just 1 column
			for (int c = 0; c < nClassIncl; ++c) {
				int[] currClsLocTest = MLFuncs.find(classLabTest, uniqClsLabTest[c]);
				int nVolsCurrClsTest = currClsLocTest.length;
				Matrix currClsCVsTest = currDimCVsTest.subMatrixRows(currClsLocTest);
	
				double currClsCVMeansTest = currClsCVsTest.colMeans()[0];
				double sqrdDistFromGrdMean = 
					Math.pow(currClsCVMeansTest - cvGrandMeansTest[d], 2);
				
				b += nVolsCurrClsTest * (sqrdDistFromGrdMean);
				
				double currClsVar = MLFuncs.var(currClsCVsTest.getColumn(0));
			    w += (nVolsCurrClsTest - 1) * currClsVar;
			}
			bwStats[d][0] = b;
			bwStats[d][1] = w;
			bwStats[d][2] = b / w;
			
			if (debug) {
				double tTime = (System.currentTimeMillis() - sTime) / 1000;
				System.out.println(" [" + tTime + "]");
			}
		
			// compute % misclassification:
			//  - classify test vols by measuring Euclidean distance from training class means
			//    and considering each test vol member of closest class using this metric
			// int nVolsTrain = cvsTrain.numRows();
			
			if (debug) {
				System.out.println("Calculating % misclassification... ");
				sTime = System.currentTimeMillis();
			}
			int[] currPredCls = new int[nVolsTest];
			
			for (int i = 0; i < nVolsTest; ++i) {
				double[][] currCVReplicated = new double[nClassIncl][1];
				for (int j = 0; j < nClassIncl; ++j) {
					currCVReplicated[j][0] = currDimCVsTest.getQuick(i, 0);
				}
				Matrix currCVRepmat = new MatrixImpl(currCVReplicated).getMatrix();
				Matrix clsDiffsFromMeanMat = (currCVRepmat.minus(
						clsMeanCVsTrain.subMatrixCols(new int[] {d})));
				double[] clsDiffs = clsDiffsFromMeanMat.getColumn(0);
				for (int k = 0; k < nClassIncl; ++k) {
					clsDiffs[k] = Math.abs(clsDiffs[k]);
				}
				double minDiff = MLFuncs.min(clsDiffs);
				if (debug) {
					System.out.println("Min class diff: " + minDiff);
					System.out.println("Class diffs: ");
					npairs.io.NpairsjIO.print(clsDiffs);
				}
				
				int minLoc = MLFuncs.find(clsDiffs, minDiff)[0]; // assign first class label satisfying
				                                                 // proximity measure  
				if (debug) {
					System.out.println("Loc of min: " + minLoc);
				}
				
				currPredCls[i] = uniqClsLabTest[minLoc];
				if (debug) {
					System.out.println("Curr predicted class: " + currPredCls[i]);
				}
			}
			
			// find percent of test vols that were misclassified
			int[] diffFromTrue = new int[nVolsTest];
			for (int i = 0; i < nVolsTest; ++i) {
				diffFromTrue[i] = currPredCls[i] - (int)currDimCVsTest.getQuick(i,0);
			}
			int[] whereTrue = MLFuncs.find(diffFromTrue, 0);
			int nWrongPred = nVolsTest - whereTrue.length;
			double pCentMisClass = (double)nWrongPred / (double)nVolsTest;
			bwStats[d][3] = pCentMisClass;
			
			if (debug) {
				double tTime = (System.currentTimeMillis() - sTime) / 1000;
				System.out.println("[" + tTime + "]");
			}
		}
		
//		// compute within-class "covariance" matrix scaledW and then invert it.
		// We are using the method in Appendix of Strother 2002 Neuroimage paper.
		
//		// TODO: Determine why idl implementation (using scaledW to weight Euclidean
//		// distances of test CV scores from training means) differs from derivation in
//		// Appendix of Strother's 2002 Neuroimage paper.
		// TODO: Determine what's going on IDL cva_wb_mats.pro (Why is returned 
		// 'within-class covariance' matrix called the 'between-class covariance matrix',
		// and moreover why is it actually apparently the identity matrix?)
		// ****CONSIDER: cv eigenvectors are normalized so each class has variance 1 
		// before data is projected onto them in order to create cv scores (see 
		// CVA.normEvectsByVar(..) )
//		Matrix W = new CVA().getW(cvsTrain, classLabTr);
//		int nVolsTr = trainVols.length;
//		Matrix scaledW = W.mult(1.0 / nVolsTr - nClassIncl);
//		Matrix invScaledW = scaledW.inverse();
		if (debug) {
			System.out.print("Calculating probs for test vols... ");
			sTime = System.currentTimeMillis();
		}
		
		// DEBUGGING
//		count = count + 1;
//		String filename = "test_scores" + count + ".txt";
//		cvsTest.printToFile(filename, "IDL");
//		filename = "train_scores" + count + ".txt";
//		cvsTrain.printToFile(filename, "IDL");		
		// END OF DEBUGGING

		
		
		for (int t = 0; t < nVolsTest; ++t) {
			
			// calculate probs for curr test vol
			double[][] probs = new double[2][nClassIncl]; // 1st row = with priors; 2nd row = no priors
			for (int c = 0; c < nClassIncl; ++c) {
				double[][] diffsFromTrMean = new double[1][nCVDims];
				for (int d = 0; d < nCVDims; ++d) {
					diffsFromTrMean[0][d] = cvsTest.getQuick(t, d) - clsMeanCVsTrain.getQuick(c, d); 
				}
				Matrix diffMat = new MatrixImpl(diffsFromTrMean).getMatrix();
//				Matrix wgtdDiffSqrd = diffMat.mult(invScaledW).mult(diffMat.transpose());
				Matrix wgtdDiffSqrd = diffMat.mult(diffMat.transpose());
				// note wgtdDiffSqrd is a grand diff metric across all cv dims, i.e., scalar
				double wgtdGrandDiffSqrd = wgtdDiffSqrd.get(0,0); 
				
				// Assume normal distribution of grand diff from mean.
				// Note Math.expm1(x) + 1 more accurate than Math.exp(x) for x near 0.
				// Using Math.exp(x) has displayed instability in test data for x near 0; switching
				// methods took care of this.  
				probs[1][c] = Math.expm1(-wgtdGrandDiffSqrd / 2) + 1;
				if (probs[1][c] == 0.0) {
					// wgtdGrandDiffSqrd so big that Math.expm1 just sets result to -1;
					// Math.exp, on the other hand, works for large negative args
					probs[1][c] = Math.exp(-wgtdGrandDiffSqrd / 2);
				}
				probs[0][c] = probs[1][c] * priors[c]; 
			}
			
			int[] loc = MLFuncs.find(probs[0], 0);
			double[] probsWithPriors = probs[0];
			if (loc.length != nClassIncl) { // if there exists at least one non-zero prob val
				// curr test vol has non-zero prob of being in at least one class
				probsWithPriors = MLFuncs.divide(probsWithPriors, MLFuncs.sum(probsWithPriors));
			}
			int[] loc2 = MLFuncs.find(probs[1], 0);
			double[] probsNoPriors = probs[1];
			if (loc2.length != nClassIncl) {
				probsNoPriors = MLFuncs.divide(probsNoPriors, MLFuncs.sum(probsNoPriors));
			}
			
			// compute some other probability stuff (using priors)
			int currClsLoc = MLFuncs.find(uniqClsLabTr, classLabTest[t])[0];    // OK because unique	
			ppTrueClass[0][t] = probsWithPriors[currClsLoc];
			ppTrueClass[1][t] = probsNoPriors[currClsLoc];
			sqrdPredError[0][t] = Math.pow(1 - probsWithPriors[currClsLoc], 2); 
			sqrdPredError[1][t] = Math.pow(1 - probsNoPriors[currClsLoc], 2);
			
			double maxProbPriors = MLFuncs.max(probsWithPriors);
			int maxLoc = MLFuncs.find(probsWithPriors, maxProbPriors)[0];
			predClass[0][t] = uniqClsLabTr[maxLoc];
			if ((int)predClass[0][t] == classLabTest[t]) {
				correctPred[0][t] = 1;
			}
			
			double maxProbNoPriors = MLFuncs.max(probsNoPriors);
			maxLoc = MLFuncs.find(probsNoPriors, maxProbNoPriors)[0];
			predClass[1][t] = uniqClsLabTr[maxLoc];
			if ((int)predClass[1][t] == classLabTest[t]) {
				correctPred[1][t] = 1;
			}
			
			int[] clsLocsTr = new int[nClassIncl]; // index locations of unique training classes 
            									   // in sorted unique class labels array
            									   // (since training classes might be subset of 
            									   // total classes)
			for (int c = 0; c < nClassIncl; ++c) {
				clsLocsTr[c] = MLFuncs.find(uniqClassLab, uniqClsLabTr[c])[0];  // OK because unique
			}	
			
			ppAllClasses[0][t] = MLFuncs.setVals(ppAllClasses[0][t], clsLocsTr, probsWithPriors);
			ppAllClasses[1][t] = MLFuncs.setVals(ppAllClasses[1][t], clsLocsTr, probsNoPriors);
			
		}
		if (debug) {
			double tTime = (System.currentTimeMillis() - sTime) / 1000;
			System.out.println("[" + tTime + "]");
		}		
	}
	
	/** Calculates mean CV score for subset of CV scores consisting of row numbers 
	 * supplied in currVolInds.  
	 * @param cvScores
	 * @param currClassLab - array (length no. of rows in cvScores) containing class 
	 * 							labels corresponding to data volumes included in cvScores  
	 * @return mean CV scores for each CV class included in current scores
	 *         - dims of output Matrix is: 
	 *           (no. included CV classes) rows X  (no. CV dims) cols
	 */

	public static Matrix getMeanCVs(Matrix cvScores, int[] currClassLab) {
		
		if (cvScores.numRows() != currClassLab.length) {
			throw new IllegalArgumentException("Incompatible sizes of input arguments");
		}
		
		int nCVDims = cvScores.numCols();
		int[] uniqClsLab = MLFuncs.sortAscending(MLFuncs.unique(currClassLab));
		int nClassIncl = uniqClsLab.length;
		
		Matrix clsMeanCVs = new MatrixImpl(nClassIncl, nCVDims).getMatrix();
		
		for (int c = 0; c < nClassIncl; ++c) {
			int[] currClsInds = MLFuncs.find(currClassLab, uniqClsLab[c]);
			Matrix currClsCVs = cvScores.subMatrixRows(currClsInds);
			double[] currClsMeans = currClsCVs.colMeans();
			clsMeanCVs.setRow(c, currClsMeans);
		}
		
		return clsMeanCVs;
	}
	
	public double[][] getPPTrueClass() {
		return ppTrueClass;
	}
	
	public double[][] getSqrdPredError() {
		return sqrdPredError;
	}
	
	public double[][] getPredClass() {
		return predClass;
	}
	
	public double[][] getCorrectPred() {
		return correctPred;
	}
	
	// return ppAllClasses for either priors or no priors
	public double[][] getPPAllClasses(int p) {
		// p: 0 == priors; 1 == no priors
		return ppAllClasses[p];
	}
}
