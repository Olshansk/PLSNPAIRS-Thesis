package resultsUtils;

import java.io.IOException;
import java.util.Arrays;
import java.util.Map;

import pls.shared.MLFuncs;

import cern.colt.matrix.linalg.Algebra;
import cern.colt.matrix.DoubleMatrix2D;

import com.jmatio.io.MatFileFilter;
import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLStructure;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLChar;
import extern.NewMatFileReader;

import npairs.io.NpairsjIO;
import npairs.shared.matlib.*;
import npairs.utils.PCA;


/** Compares PLS and NPAIRS result eigenimages. 
 * 	(NPAIRS: Z-scored average eigenimages for each CV dim;
 * 		PLS: original brainlvs for each LV dim (NOT the bootstrap
 *           ratio results
 *           TODO: find out whether it should be the BS ratio results
 *                 or some other images that should be compared instead
 *                 with the NPAIRS results) 
 * Comparison is done by doing PCA separately on both PLS and NPAIRS images
 * and then correlating resulting PCA eigenimages from PLS with PCA eigenimages
 * from NPAIRS.  Note that each PLS lag is dealt with separately, i.e., an 
 * individual PCA is done for each PLS lag and the lag results compared with 
 * NPAIRS.
*/

public class PCAPlsVsNpairs {
	
//	String plsPCA;
//	String npairsPCA;
	
	int plsMskSize;
	int npairsMskSize;
		
	public PCAPlsVsNpairs(String plsFile, String npairsFile) throws IOException, MatrixException {
		
		// load PLS and NPAIRS images
		Map<String, MLArray> plsResultInfo = null;
		plsResultInfo = new NewMatFileReader(plsFile,
				new MatFileFilter(new String[]{"st_coords", "brainlv", "boot_result", 
						"st_win_size"}) ).getContent();
		Map<String, MLArray> npairsResultInfo = null;
		npairsResultInfo = new NewMatFileReader(npairsFile,
					new MatFileFilter(new String[]{"st_coords",
							"npairs_result"}) ).getContent();
		
		//MLDouble plsImgs = (MLDouble)plsResultInfo.get("brainlv");
		// load bootstrap:
		MLStructure bootResult = (MLStructure)plsResultInfo.get("boot_result");
		MLDouble plsImgs  = (MLDouble)bootResult.getField("compare");
		
		MLStructure npairsResult = (MLStructure) npairsResultInfo.get("npairs_result");
		MLDouble npairsImgs = (MLDouble)npairsResult.getField("zscored_brainlv_avg");

		int plsImgSize = plsImgs.getM();
		int plsNLVDims = plsImgs.getN();
		
		int npairsImgSize = npairsImgs.getM();
		int npairsNCVDims = npairsImgs.getN();
		System.out.println("Size of pls LV imgs: " + plsImgSize + " X " + plsNLVDims);
		System.out.println("Size of npairs CV imgs: " + npairsImgSize + " X " + 
				npairsNCVDims);
		
		// getMask sets mask size too
		int[] plsMskCoords = getMask(plsResultInfo, "PLS");
		int[] npairsMskCoords = getMask(npairsResultInfo, "NPAIRS");
		System.out.println("Size of PLS mask: " + plsMskSize);
		System.out.println("Size of NPAIRS mask: " + npairsMskSize);	
		
		if (!Arrays.equals(plsMskCoords, npairsMskCoords)) {
			System.out.println("PLS and NPAIRS have different mask coords!");
			System.exit(1);
		}
		if (npairsMskSize != npairsImgSize) {
			System.out.println("npairs img size and mask size are incompatible!");
			System.exit(1);
		}
		// save mask indices
//	    NpairsjIO.printToIDLFile(npairsMskCoords,"mask_coords.1D");
		
		int plsWinSize = ((MLDouble)plsResultInfo.get("st_win_size")).get(0).intValue();
		System.out.println("PLS window size: " + plsWinSize);
		if ((plsImgSize / plsWinSize) != plsMskSize) {
			System.out.println("pls img size and mask size are incompatible!");
			System.exit(1);
		}
		
		// Do PCA for NPAIRS
		Matrix npairsImgMat = new MatrixImpl(npairsImgs.getArray(), "COLT").getMatrix();
		System.out.println("Size of npairs img mat Matrix being fed into PCA: " + 
				npairsImgMat.numRows() + " X " + npairsImgMat.numCols());
		PCA npairsPCA = new PCA(npairsImgMat.transpose(), false, false);
		Matrix npairsPCAEvects = npairsPCA.getEvects();
		System.out.println("Size npairs PCA evects Matrix: " + 
				npairsPCAEvects.numRows() + " X " + npairsPCAEvects.numCols());
		double minNpairsPCs = MLFuncs.min(npairsPCAEvects.toRowPacked1DArray());
		double maxNpairsPCs = MLFuncs.max(npairsPCAEvects.toRowPacked1DArray());
		System.out.println("min NPAIRS pcs: " + minNpairsPCs);
		System.out.println("max NPAIRS pcs: " + maxNpairsPCs);
		
		// Save NPAIRS PCs:
//		npairsPCAEvects.printToFile("npairsPCEvects.2D", "IDL");
		
		// For each lag, get PLS images for each condition 
		Matrix plsImgsMat = new MatrixImpl(plsImgs.getArray(), 
				"COLT").getMatrix();		
		
		double minPlsImgsMat = MLFuncs.min(plsImgsMat.toRowPacked1DArray());
		double maxPlsImgsMat = MLFuncs.max(plsImgsMat.toRowPacked1DArray());
		System.out.println("Min PLS Imgs mat: " + minPlsImgsMat);
		System.out.println("Max PLS Imgs mat: " + maxPlsImgsMat);
		
		
		Matrix[] corrCoeffs = new Matrix[plsWinSize];
		
		// rearrange PLS matrix for all lags (to do all-lag PCA): each lag is treated as separate
		// image dimension 
		Matrix reArrangedPLSImgsAllLags = new MatrixImpl(plsMskSize, plsNLVDims*plsWinSize).getMatrix();
		for (int i = 0; i < plsWinSize; ++i) {		  
			int[] currLagRows = new int[plsMskSize];
			System.out.println("Curr lag rows: ");
			for (int j = 0; j < plsMskSize; ++j) {
				currLagRows[j] = (plsWinSize * j) + i;
				if (j < 20) {
					System.out.print(currLagRows[j] + " ");
				}
			}
			System.out.println();
//			NpairsjIO.printToIDLFile(currLagRows,"rows_lag" + i + ".1D");

			Matrix currLagLVs = plsImgsMat.subMatrixRows(currLagRows);
			// add curr lag to rearranged PLS matrix
			reArrangedPLSImgsAllLags.setSubMatrix(currLagLVs, 0, 0 + (i * plsNLVDims));
			//			currLagLVs.printToFile("LV_lag" + i + ".2D", "IDL");

			//			currLagLVs = currLagLVs.subMatrix(new int[]{0, plsMskSize-1}, 
			//					new int[]{0, plsNLVDims - 2});
			double minCurrLVMat = MLFuncs.min(currLagLVs.toRowPacked1DArray());
			double maxCurrLVMat = MLFuncs.max(currLagLVs.toRowPacked1DArray());
			System.out.println("Min Curr Lag LV mat: " + minCurrLVMat);
			System.out.println("Max Curr Lag LV mat: " + maxCurrLVMat);

			// correlate current lag LVs with npairs CVs
			corrCoeffs[i] = currLagLVs.correlate(npairsImgMat);
			System.out.println("Correlation of Lag # " + i + " LVs with " +
					"NPAIRS CVs: ");
			corrCoeffs[i].print();
			
			// don't normalize PCs by SD to have variance 1
			// (They will have length 1 instead)
			System.out.println("Size of pls img mat Matrix being fed into PCA " +
					"for Lag # " + i + ": " + currLagLVs.numRows() + " X " +
					currLagLVs.numCols());
			// Do PCA separately for each lag
			PCA plsPCA = new PCA(currLagLVs.transpose(), false, false);
			Matrix PCAEvects = plsPCA.getEvects();
			System.out.println("Size pls PCA evects Matrix: " + 
					PCAEvects.numRows() + " X " + PCAEvects.numCols());
			double minPLSPCs = MLFuncs.min(PCAEvects.toRowPacked1DArray());
			double maxPLSPCs = MLFuncs.max(PCAEvects.toRowPacked1DArray());
			System.out.println("min PLS pcs: " + minPLSPCs);
			System.out.println("max PLS pcs: " + maxPLSPCs);

			double[] pcEvals = plsPCA.getEvals();
			System.out.println("PLS evals: ");
			NpairsjIO.print(pcEvals);
		// Compare current lag PCA eigenimages with NPAIRS PCA eigenimages:

			// Save pc matrices:
			PCAEvects.printToFile("plsPCEvects_LV" + i + ".2D", "IDL");

			// (i) Correlation coeffs:
			Matrix corrMat = PCAEvects.correlate(npairsPCAEvects);
			System.out.println("Correlation matrix: ");
			corrMat.print();

			// pcaEvects = npairsEvects * B + E
			Algebra alg = new Algebra();
			try {
				DoubleMatrix2D B = alg.solve(((ColtMatrix)npairsPCAEvects).getColtMat(),((ColtMatrix)PCAEvects).getColtMat());
				Matrix Bmat = new MatrixImpl(B.toArray()).getMatrix();

				// E = plsEvects - npairssEvects*B
				Matrix Emat = PCAEvects.minus(npairsPCAEvects.mult(Bmat));
				System.out.println("LAG # " + i); 
				System.out.println("solution to P = N*B + E: ");
				System.out.println("B size: " + Bmat.numRows() + " X " +
						Bmat.numCols());
				System.out.println("B: ");
				Bmat.print();
				System.out.println("E size: " + Emat.numRows() + " X " +
						Emat.numCols());
				//				System.out.println("E: ");
				//				Emat.print();

			}
			catch (IllegalArgumentException iae) {
				System.out.println("Unable to solve N = P*X for Lag # " + i + ": " + iae.getMessage());
			}
		}
		// Do PCA on all-lag PLS LVs:
		System.out.println("Size of rearranged PLS imgs: " + reArrangedPLSImgsAllLags.numRows() +
				" X " + reArrangedPLSImgsAllLags.numCols());
		
		System.out.println("Doing PCA on all PLS lags at once...");
		PCA allLagPLSPCA = new PCA(reArrangedPLSImgsAllLags.transpose(), false, false);
		System.out.println("Done all-lag PCA.");
		Matrix allLagPCAEvects = allLagPLSPCA.getEvects();
		System.out.println("Size pls all-lag PCA evects Matrix: " + 
				allLagPCAEvects.numRows() + " X " + allLagPCAEvects.numCols());
		double minPLSPCs = MLFuncs.min(allLagPCAEvects.toRowPacked1DArray());
		double maxPLSPCs = MLFuncs.max(allLagPCAEvects.toRowPacked1DArray());
		System.out.println("min PLS pcs: " + minPLSPCs);
		System.out.println("max PLS pcs: " + maxPLSPCs);

		double[] pcEvals = allLagPLSPCA.getEvals();
		System.out.println("PLS evals (normalized so sum of evals is 1: ");
		double sumEvals = MLFuncs.sum(pcEvals);
		for (int i = 0; i < pcEvals.length; ++i) {
			pcEvals[i] = pcEvals[i] / sumEvals;
		}
		System.out.println("Sum evals: " + MLFuncs.sum(pcEvals));
		NpairsjIO.print(pcEvals);
	// Compare current lag PCA eigenimages with NPAIRS PCA eigenimages:

		// Save pc matrices 
		allLagPCAEvects.printToFile("plsPCEvects_allLag.2D", "IDL");

		// (i) Correlation coeffs:
		Matrix corrMat = allLagPCAEvects.correlate(npairsPCAEvects);
		System.out.println("Correlation matrix between all-lag PC evects and NPAIRS PC evects: ");
		corrMat.print();
	
	}



	private int[] getMask(Map<String, MLArray> resultInfo, String resultType) {
		MLDouble coordML = (MLDouble)resultInfo.get("st_coords");
		int mskSize = coordML.getSize();
		int[] maskCoords = new int[mskSize];
		for (int i = 0; i < mskSize; ++i) {
			maskCoords[i] = coordML.get(i).intValue() - 1; // we want 0-relative coords
		}
		if (resultType.equals("PLS")) {
			plsMskSize = mskSize;
		}
		else npairsMskSize = mskSize;
		return maskCoords;
	}
		
		 
		
		public static void main(String[] args) {
			try {
				PCAPlsVsNpairs pca = new PCAPlsVsNpairs(args[0], args[1]);
			}
			catch (Exception ioe) {
				ioe.printStackTrace();
			}
		}
		
}
