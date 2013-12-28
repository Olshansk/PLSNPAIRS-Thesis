package npairs.utils;

import pls.shared.MLFuncs;
import npairs.shared.matlib.*;

//for testing:
import npairs.io.*;
import java.io.IOException;

public class ZScorePatternInfo {
	
	boolean debug = false;
	
	Matrix sigPattern;   // projection of input patterns onto signal axis
	
	Matrix noisePattern; // projection of input patterns onto noise axis
	
	private double[] noisePattVar; // variance of noise pattern
	
	double[] noisePattStdDev; // standard deviation of noise pattern;
	
	
	
	double[] r;     // Pearson's corr. coeffs. between input patterns 
					// (1 r-value for each dim (col) of input patterns))
	
	/** Constructor
	 * 
	 * @param patt1
	 * @param patt2
	 */
	public ZScorePatternInfo(Matrix patt1, Matrix patt2) {
		projectZScoredData(patt1, patt2);
	}
	
	
	/** Returns rSPM(Z) for given input patterns
	 * 
	 * @param patt1 
	 * 			rows = voxels
	 *          cols = model dims; each dim has separate pattern
	 *
	 * @param patt2
	 *          same dims as patt1
	 * @return Matrix 
	 * 			same dims as input patterns
	 * 
	 */ 
	private void projectZScoredData(Matrix pattern1, Matrix pattern2) {
		// Calculate proj. onto sig. axis (z1 + z2)/ sqrt(2) 
		// and proj. onto noise axis (z1 - z2) / sqrt(2),
		// where z1 = patt1 - mean(patt1)/stddev(patt1)
		// and similarly for z2.
		// Divide projections by stddev(noise axis).
		// Calculate normalized sig-axis proj. ( == rSPM(Z))
		// (and also normalized noise-axis proj.)
		//
		
		// DEBUGGING
//		pattern1.printToFile("pattern1.txt", "IDL");
//		pattern2.printToFile("pattern2.txt", "IDL");
		// END OF DEBUGGING
		
		Matrix patt1 = pattern1.copy();
		Matrix patt2 = pattern2.copy();
		int numRows = patt1.numRows();
		int numCols = patt1.numCols();
		if ((numRows != patt2.numRows()) || numCols != patt2.numCols()) {
			throw new IllegalArgumentException("Input Matrices must have the same dimensions");
		}
		
		// mean-centre and divide each input data col by its std dev
		// normedMeans1, normedMeans2 are patt1,patt2 means divided
		// by corresp. patt stddev.
		double[] normedMeans1 = patt1.zScoreCols();
		double[] normedMeans2 = patt2.zScoreCols();
		if (debug) {
			System.out.println("NormedMeans1: ");
			NpairsjIO.print(normedMeans1);
			System.out.println("NormedMeans2: ");
			NpairsjIO.print(normedMeans2);
		}
		
		// project onto sig and noise axes
		sigPattern = (patt1.plus(patt2)).mult(1 / Math.sqrt(2));
		noisePattern = (patt1.minus(patt2)).mult(1 / Math.sqrt(2));
	
		noisePattVar = new double[numCols];
		noisePattStdDev = new double[numCols];
		r = new double[numCols];
		// find noise-axis standard deviation for each column [model dim]
		for (int c = 0; c < numCols; ++c) {
			double[] currNoiseData = noisePattern.getColumnQuick(c);
			noisePattVar[c] = MLFuncs.var(currNoiseData);
			noisePattStdDev[c] = Math.sqrt(noisePattVar[c]);
			double[] currSigData = sigPattern.getColumnQuick(c);	
			// project means for curr patt col onto sig and
			// noise axes and normalize by noise std dev
			double sigMean = (normedMeans1[c] * (1 / Math.sqrt(2))) +
			 (normedMeans2[c] * (1 / Math.sqrt(2)));
			double noiseMean = (normedMeans1[c] * (1 / Math.sqrt(2))) - 
			 (normedMeans2[c] * (1 / Math.sqrt(2)));
			sigMean /= noisePattStdDev[c];
			noiseMean /= noisePattStdDev[c];
			if (debug) {
				System.out.println("SigMean after div by noisePattStdDev: " + sigMean);
				System.out.println("NoiseMean after div by noisePattStdDev: " + noiseMean);
			}
			for (int i = 0; i < numRows; ++i) {
				currSigData[i] /= noisePattStdDev[c];
				currNoiseData[i] /= noisePattStdDev[c];
				// add back the means
				currSigData[i] += sigMean;
				currNoiseData[i] += noiseMean;
			}		
			sigPattern.setColumnQuick(c, currSigData);
			noisePattern.setColumnQuick(c, currNoiseData);		
			
			r[c] = 1 - noisePattVar[c];
		}		
	}
	
	public Matrix getSignalPattern() {
		return sigPattern;
	}
	
	public Matrix getNoisePattern() {
		return noisePattern;
	}
	
	public double[] getCorrCoeffs() {
		return r;
	}
	
	public double[] getNoiseStdDev() {
		return noisePattStdDev;
	}
	
//	/** Returns covariance of 2 inpt arrays of data
//	 *  REQUIRED: input arrays are of same length
//	*/
//	private double cov(double[] vec1, double[] vec2, boolean meanCentred) {
//		double cov = 0;
//		double avg1 = 0;
//		double avg2 = 0;
//		if (!meanCentred) {
//			avg1 = MLFuncs.avg(vec1);
//			avg2 = MLFuncs.avg(vec2);		
//		}
//		
//		for (int i = 0; i < vec1.length; ++i) {
//			cov += (vec1[i] - avg1) * (vec2[i] - avg2);
//		}
//		
//		cov /= (vec1.length - 1);
//		return cov;
//	}

	// For testing - args[0] = matlibType
	// 
	public static void main (String[] args) {
		
		String[] dataFilenames = new String[2];
		dataFilenames[0] = args[0];
		dataFilenames[1] = args[1];
		
		String[] maskFilenames = new String[2];
		maskFilenames[0] = args[2];
		maskFilenames[1] = args[2];
		
		try {
			double[][] data = NiftiIO.readNiftiData(dataFilenames, maskFilenames);
			MatrixImpl mImpl = new MatrixImpl(data, "COLT");
			Matrix dataMat = mImpl.getMatrix();
			Matrix patt1 = dataMat.subMatrixRows(new int[] {0}).transpose();
//			System.out.println("Size of patt1: " + patt1.numRows() + " rows X " + patt1.numCols() + " cols");
			Matrix patt2 = dataMat.subMatrixRows(new int[] {1}).transpose();
//			System.out.println("Size of patt2: " + patt2.numRows() + " rows X " + patt2.numCols() + " cols");
			System.out.println("Begin ZScorePatternInfo...");
			ZScorePatternInfo zsPattInfo = new ZScorePatternInfo(patt1, patt2);
			System.out.println("End ZScorePatternInfo...");
			Matrix sigPatt = zsPattInfo.getSignalPattern();
			Matrix noisePatt = zsPattInfo.getNoisePattern();
			double[] CC = zsPattInfo.getCorrCoeffs();
		//	System.out.println("CC: ");
			//NpairsjIO.print(CC);
			double[] noiseSD = zsPattInfo.getNoiseStdDev();
		//	System.out.println(" noise SD: ");
			//NpairsjIO.print(noiseSD);
		//  System.out.println("1st 10 els of sig patt:" +
		//			" ");
			//NpairsjIO.print(sigPatt.subMatrix(new int[] {0, 9}, new int[] {0, 0}).toArray());
		//	System.out.println("1st 10 els of noise patt: ");
			//NpairsjIO.print(noisePatt.subMatrix(new int[] {0, 9}, new int[] {0, 0}).toArray());
			
		
			
			
			
		}
		catch(MatrixException me) {
			me.printStackTrace();
		}
		catch (IOException e) {
			e.printStackTrace();
		}		
	}
	
}