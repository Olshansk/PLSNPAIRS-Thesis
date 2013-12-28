package npairs.io;

//import npairs.shared.matlib.*;
import java.io.*;
import java.util.*;

import com.jmatio.io.MatFileFilter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLStructure;

import npairs.NpairsjException;
import npairs.NpairsjSetupParams;
import npairs.shared.matlib.Matrix;
import npairs.shared.matlib.MatrixException;
import npairs.shared.matlib.MatrixImpl;
import extern.NewMatFileReader;
import extern.niftijlib.*;
//import npairs.NpairsjException;
import pls.shared.MLFuncs;
import npairs.io.NpairsjIO;

/** Convenience class of Nifti IO methods for reading/writing Nifti data.
 * Uses Nifti1Dataset.
 * 
 * @author anita
 *
 */
public class NiftiIO {

	final static boolean debug = false;
	// Valid DATA_TYPES are specified in nifti-1 documentation; see
	// http://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h
	final static int[] DATA_TYPES = { 0, 1, 2, 4, 8, 16, 32, 64, 128, 255, 
								256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304 }; 
	
	
	/** Writes 3D volume into nifti file 'filename' using Nifti1Dataset.  
	 *  File is of .nii format iff filename contains .nii extension; otherwise
	 *  data is saved into pair of files with .img/.hdr extensions.
	 *  
	 * @param vol3D - data to be saved as doubles; data must be in [Z][Y][X] order
	 * @param datatype - 
	 *             Edited description from http://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h:
	 *          
     *              /*--- the original ANALYZE 7.5 type codes ---*
	                 				 
                  	0     /* [unknown]         *
                  	1     /* binary (1 bit/voxel)         *
          		 	2     /* unsigned char (8 bits/voxel) *
          			4     /* signed short (16 bits/voxel) *
           			8     /* signed int (32 bits/voxel)   *
                 	16    /* float (32 bits/voxel)        *
             		32    /* complex (64 bits/voxel)      *
                 	64    /* double (64 bits/voxel)       *
                   	128   /* RGB triple (24 bits/voxel)   *
                	255   /* not very useful (?)          *

 
                          /*------------------- new codes for NIFTI ---*
                 	256   /* signed char (8 bits)         *
              		512   /* unsigned short (16 bits)     *
              		768   /* unsigned int (32 bits)       *
               		1024  /* long long (64 bits)          *
             		1280  /* unsigned long long (64 bits) *
            		1536  /* long double (128 bits)       *
         			1792  /* double pair (128 bits)       *
        			2048  /* long double pair (256 bits)  *
              		2304  /* 4 byte RGBA (32 bits/voxel)  *
     * @param voxSize 3-element double array containing voxel size of data {Xsize, Ysize, Zsize}
	 * @param filename
	 * @throws IOException
	 * @throws FileNotFoundException
	 * @see #writeVol(double[], int[], int, double[], String)
	 * @see #writeVol(double[], int[], double[], String)
	 * @see #writeVol(double[], int[], boolean, int[], double[], String)
	 * @see #writeVol(double[][][], String)
	 * @see #writeVol(double[][][], int, String)
	 * @see #writeVol(double[][][], double[], String)
	 * 

	 */
	public static void writeVol(double[][][] vol3D, int datatype, double[] voxSize, String filename) throws IOException, FileNotFoundException {
		Nifti1Dataset niftiDS = new Nifti1Dataset();
		niftiDS.setHeaderFilename(filename);
		niftiDS.setDataFilename(filename);
		if (!MLFuncs.contains(DATA_TYPES, datatype)) {
			throw new IllegalArgumentException("Invalid datatype: " + datatype);
		}
		niftiDS.setDatatype((short)datatype);
		niftiDS.setVoxSize(voxSize);
		niftiDS.setDims((short)3, (short)vol3D[0][0].length, (short)vol3D[0].length, (short)vol3D.length, 
				(short)0, (short)0, (short)0, (short)0);
		niftiDS.writeHeader();
		niftiDS.writeVol(vol3D, (short)0);
	}
	
	/** Writes 3D volume into nifti file 'filename' using Nifti1Dataset.  
	 *  File is of .nii format iff filename contains .nii extension; otherwise
	 *  data is saved into pair of files with .img/.hdr extensions. 
	 *  Voxel size is set to default 1mm^3.  
	 * @param vol3D - data to be saved as doubles; data must be in [Z][Y][X] order
	 * @param datatype - see writeVol(double[][][], int, double[], String) for details
	 * @param filename 
	 * @throws IOException
	 * @throws FileNotFoundException
	 * @see #writeVol(double[], int[], int, double[], String)
	 * @see #writeVol(double[], int[], double[], String)
	 * @see #writeVol(double[], int[], boolean, int[], double[], String)
	 * @see #writeVol(double[][][], String)
	 * @see #writeVol(double[][][], int, double[], String)
	 * @see #writeVol(double[][][], double[], String)
	 */
	public static void writeVol(double[][][] vol3D, int datatype, String filename) throws IOException, 
		FileNotFoundException {
		// use default voxel size {1, 1, 1};
		double[] defaultVoxSz = {1, 1, 1};
		writeVol(vol3D, datatype, defaultVoxSz, filename);
	}

	/** Writes 3D volume into nifti file 'filename' using Nifti1Dataset.  
	 *  File is of .nii format iff filename contains .nii extension; otherwise
	 *  data is saved into pair of files with .img/.hdr extensions.
	 *  Data is saved as floats (32 bits / voxel)
	 *  Voxel size is set to default 1mm^3.
	 *  
	 * @param vol3D - data to be saved; data must be in [Z][Y][X] order
	 * @param filename
	 * @throws IOException
	 * @throws FileNotFoundException
	 * @see #writeVol(double[], int[], int, double[], String)
	 * @see #writeVol(double[], int[], double[], String)
	 * @see #writeVol(double[], int[], boolean, int[], double[], String)
	 * @see #writeVol(double[][][], int, String)
	 * @see #writeVol(double[][][], int, double[], String)
	 * @see #writeVol(double[][][], double[], String)
	 */
	public static void writeVol(double[][][] vol3D, String filename) throws IOException, FileNotFoundException {
		writeVol(vol3D, 16, filename);
	}
	
	/** Writes 3D volume into nifti file 'filename' using Nifti1Dataset.  
	 *  File is of .nii format iff filename contains .nii extension; otherwise
	 *  data is saved into pair of files with .img/.hdr extensions.
	 *  Data is saved as floats (32 bits / voxel) 
	 *  
	 * @param vol3D - data to be saved; data must be in [Z][Y][X] order
	 * @param voxSize - 3-element double array containing voxel size of data {Xsize, Ysize, Zsize}
	 * @param filename
	 * @throws IOException
	 * @throws FileNotFoundException
	 * @see #writeVol(double[], int[], int, double[], String)
	 * @see #writeVol(double[], int[], double[], String)
	 * @see #writeVol(double[], int[], boolean, int[], double[], String)
	 * @see #writeVol(double[][][], int, String)
	 * @see #writeVol(double[][][], int, double[], String)
	 * @see #writeVol(double[][][], String)
	 * 
	 */
	public static void writeVol(double[][][] vol3D, double[] voxSize, String filename) throws IOException, 
		FileNotFoundException {
		writeVol(vol3D, 16, voxSize, filename);
	}
	
	
	/** Writes 1D volume into 3d nifti file 'filename' using Nifti1Dataset.  
	 *  File is of .nii format iff filename contains .nii extension; otherwise
	 *  data is saved into pair of files with .img/.hdr extensions.
	 *  
	 * @param vol1D - data to be saved; data must be in xyz order, i.e.,
	 *              data[x + y*xdim + *xdim*ydim] = data3D[z][y][x]
	 * @param volDims - 3-element int array containing dimensions of volume to be saved
	 *                  {xdim, ydim, zdim}
	 * @param datatype -
	 *             Edited description from http://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h:
	 *          
     *              /*--- the original ANALYZE 7.5 type codes ---*
	                 				 
                  	0     /* [unknown]         *
                  	1     /* binary (1 bit/voxel)         *
          		 	2     /* unsigned char (8 bits/voxel) *
          			4     /* signed short (16 bits/voxel) *
           			8     /* signed int (32 bits/voxel)   *
                 	16    /* float (32 bits/voxel)        *
             		32    /* complex (64 bits/voxel)      *
                 	64    /* double (64 bits/voxel)       *
                   	128   /* RGB triple (24 bits/voxel)   *
                	255   /* not very useful (?)          *

 
                          /*------------------- new codes for NIFTI ---*
                 	256   /* signed char (8 bits)         *
              		512   /* unsigned short (16 bits)     *
              		768   /* unsigned int (32 bits)       *
               		1024  /* long long (64 bits)          *
             		1280  /* unsigned long long (64 bits) *
            		1536  /* long double (128 bits)       *
         			1792  /* double pair (128 bits)       *
        			2048  /* long double pair (256 bits)  *
              		2304  /* 4 byte RGBA (32 bits/voxel)  *
     * @param voxSize - 3-element double array containing voxel size of data {Xsize, Ysize, Zsize}
     * @param origin - 3-element int array containing location of origin in voxel coordinates
	 * @param filename
	 * @throws IOException
	 * @throws FileNotFoundException
	 * @see #writeVol(double[][][], String)
	 * @see #writeVol(double[][][], int, String)
	 * @see #writeVol(double[][][], int, double[], String)
	 * @see #writeVol(double[][][], double[], String)
	 * @see #writeVol(double[], int[], double[], String)
	 * @see #writeVol(double[], int[], boolean, int[], double[], String)
	 */
	public static void writeVol(double[] vol1D, int[] volDims, int datatype, double[] voxSize,
			int[] origin, String filename) throws IOException, FileNotFoundException {
		Nifti1Dataset niftiDS = new Nifti1Dataset();
		niftiDS.setHeaderFilename(filename);
		niftiDS.setDataFilename(filename);
		if (!MLFuncs.contains(DATA_TYPES, datatype)) {
			throw new IllegalArgumentException("Invalid datatype: " + datatype);
		}
		niftiDS.setDatatype((short)datatype);
		niftiDS.setVoxSize(voxSize);
		niftiDS.setDims((short)3, (short)volDims[0], (short)volDims[1], (short)volDims[2],
				(short)0, (short)0, (short)0, (short)0);
		float[] voxOffset = new float[] {(float)-22, (float)31, (float)18};
//		float mmOriginX = (float)((Math.abs(voxOffset[0]) + 0.5) * (Math.signum(voxOffset[0])*voxSize[0]));
//		float mmOriginY = (float)((Math.abs(voxOffset[1]) + 0.5) * (Math.signum(voxOffset[1])*voxSize[1])); 
//		float mmOriginZ = (float)((Math.abs(voxOffset[2]) + 0.5) * (Math.signum(voxOffset[2])*voxSize[2])); 
		float mmOriginX = (float)((Math.abs(voxOffset[0]) + 0.5) * voxSize[0]);
		float mmOriginY = (float)((Math.abs(voxOffset[1]) + 0.5) * voxSize[1]); 
    	float mmOriginZ = (float)((Math.abs(voxOffset[2]) + 0.5) * voxSize[2]); 
		niftiDS.qoffset = new float[] {-mmOriginX, -mmOriginY, -mmOriginZ};
		niftiDS.quatern = new float[] {(float)0.0, (float)1.0, (float)0.0};
		niftiDS.srow_x = new float[] {(float)voxSize[0], (float)0.0, (float)0.0, -mmOriginX};
		niftiDS.srow_y = new float[] {(float)0.0, (float)voxSize[1], (float)0.0, -mmOriginY};
		niftiDS.srow_z = new float[] {(float)0.0, (float)0.0, (float)voxSize[2], -mmOriginZ};
		niftiDS.sform_code = 2;
		niftiDS.qform_code = 4;
		
		niftiDS.qfac = -1;
		
		niftiDS.writeHeader();
		niftiDS.printHeader();
		double[][][] vol3D = MLFuncs.reshapeZYX(vol1D, volDims[0], volDims[1], volDims[2]);
		niftiDS.writeVol(vol3D, (short)0);
	}
	
	/** Just for testing loadBgImg... */
	public static void writeVol(double[] vol1D, String filename)
		throws IOException, FileNotFoundException {
		writeVol(vol1D, new int[] {64, 70, 54}, 16, new double[] {3.125, 3.125, 3.125}, null, 
				filename);
	}
	
	/** Writes 1D volume into 3d nifti file 'filename' using Nifti1Dataset.  
	 *  File is of .nii format iff filename contains .nii extension; otherwise
	 *  data is saved into pair of files with .img/.hdr extensions.
	 *  Data is saved as floats (32 bits / voxel)
	 *  
	 * @param vol1D - data to be saved; data must be in xyz order, i.e.,
	 *              data[x + y*xdim + *xdim*ydim] = data3D[z][y][x]
	 * @param volDims - 3-element int array containing dimensions of volume to be saved
	 *                  {xdim, ydim, zdim}
	 * @param voxSize - 3-element double array containing voxel size of data {Xsize, Ysize, Zsize}
	 * @param origin - 3-element int array containing location of origin in voxel coordinates
	 * @param filename
	 * @throws IOException
	 * @throws FileNotFoundException
	 * @see #writeVol(double[][][], String)
	 * @see #writeVol(double[][][], int, String)
	 * @see #writeVol(double[][][], int, double[], String)
	 * @see #writeVol(double[][][], double[], String)
	 * @see #writeVol(double[], int[], int, double[], String)
	 * @see #writeVol(double[], int[], boolean, int[], double[], String)
	 */
	public static void writeVol(double[] vol1D, int[] volDims, double[] voxSize, int[] origin,
			String filename) throws IOException, FileNotFoundException {
		writeVol(vol1D, volDims, 16, voxSize, origin, filename);
	}
	
	/** Writes masked 1D volume into 3D nifti file 'filename' using Nifti1Dataset. 
	 *  File is of .nii format iff filename contains .nii extension; otherwise
	 *  data is saved into pair of files with .img/.hdr extensions.
	 *  Data is saved as floats (32-bits/ voxel)
	 *  
	 *  @param vol1D - data to be saved; data is in same order as in maskInds as
	 *               each element in data is mapped to corresponding maskInds 
	 *               location in 3D volume
	 *  @param maskInds - indices in 3D volume of elements stored in vol1D 
	 *                  - indices are in xyz order; the 3D 
	 *                    volume will be in [Z][Y][X] order
	 * @param oneRel - true if maskInds are 1-relative; false if they are 0-relative
	 * @param volDims - 3-element int array containing dimensions of volume to be saved
	 *                  {xdim, ydim, zdim}
	 * @param voxSize - 3-element double array containing voxel size of data {Xsize, Ysize, Zsize}
	 * @param origin - 3-element int array containing location of origin in voxel coordinates
	 * @param filename
	 * @throws IOException
	 * @throws FileNotFoundException
	 * @see #writeVol(double[][][], String)
	 * @see #writeVol(double[][][], int, String)
	 * @see #writeVol(double[][][], int, double[], String)
	 * @see #writeVol(double[][][], double[], String)
	 * @see #writeVol(double[], int[], int, double[], String)
	 * @see #writeVol(double[], int[], double[], String)
	 */
	public static void writeVol(double[] vol1D, int[] maskInds, boolean oneRel, int[] volDims, 
			double[] voxSize, int[] origin, String filename) throws IOException, FileNotFoundException {
		if (vol1D.length != maskInds.length) {
			throw new IllegalArgumentException("Data and mask index arrays must " +
					"be of same length.");
		}
		if (volDims.length != 3) {
			throw new IllegalArgumentException("Volume dimension info must have 3 elements.");
	
		}
		if (voxSize.length != 3) {
			throw new IllegalArgumentException("Voxel size info must have 3 elements.");
		}
		if (origin.length != 3) {
			throw new IllegalArgumentException("Origin info must have 3 elements.");
		}

		int nVox = volDims[0] * volDims[1] * volDims[2];
		
		double[] fullVol1D = new double[nVox]; // embed input data into 1D vol 
		if (oneRel) {
			for (int i = 0; i < maskInds.length; ++i) {
				maskInds[i]--;
			}
		}
		for (int i = 0; i < maskInds.length; ++i) {
			int currIdx = maskInds[i];
			if (currIdx < 0 || currIdx >= nVox) {
				throw new IllegalArgumentException("Invalid mask index element: " + 
						currIdx + ".  Largest valid value: " + (nVox - 1));
			}
			fullVol1D[currIdx] = vol1D[i];
		}
		writeVol(fullVol1D, volDims, voxSize, origin, filename);
	}
	
	public static void writeVol4DNpairs(double[][] cvs, int[] maskInds,
			boolean oneRel, int[] volDims, double[] voxSize, String filename)
			throws IOException, FileNotFoundException{
		
		if (cvs[0].length != maskInds.length) {
			throw new IllegalArgumentException("Data and mask index arrays must " +
					"be of same length.");
		}
		if (volDims.length != 3) {
			throw new IllegalArgumentException("Volume dimension info must have 3 elements.");
	
		}
		if (voxSize.length != 3) {
			throw new IllegalArgumentException("Voxel size info must have 3 elements.");
		}

		if (oneRel) {
			for (int i = 0; i < maskInds.length; ++i) {
				maskInds[i]--;
			}
		}
		
		int nVox = volDims[0] * volDims[1] * volDims[2];
		double[][] fullVol2D = new double[cvs.length][nVox]; // embed input data into 1D vol
		for (int i = 0; i < maskInds.length; ++i) {
			int currIdx = maskInds[i];
			if (currIdx < 0 || currIdx >= nVox) {
				throw new IllegalArgumentException("Invalid mask index element: " + 
						currIdx + ".  Largest valid value: " + (nVox - 1));
			}
			for(int cv = 0; cv < cvs.length; cv++){
				fullVol2D[cv][currIdx] = cvs[cv][i]; 
			}
		}
		writeVol4DNiftiDataset(volDims, voxSize, cvs.length, filename, fullVol2D);
	}
	
	//input lags should be zero relative.
	public static void writeVol4DPLS(double[] singleLV, int[] maskInds,boolean oneRel, 
			int winSize, int[] lags, int[] volDims, double[] voxSize, String filename) 
			throws IOException, FileNotFoundException{
		
		if(singleLV.length / maskInds.length != winSize){
			throw new IllegalArgumentException("Data and mask index arrays must " +
			"be such that len(data) / len(mask) = number of lags");
		}

		if(lags.length == 0){
			throw new IllegalArgumentException("No lags specified to extract.");
		}
		
		if (volDims.length != 3) {
			throw new IllegalArgumentException("Volume dimension info must have 3 elements.");
	
		}
		if (voxSize.length != 3) {
			throw new IllegalArgumentException("Voxel size info must have 3 elements.");
		}
		
		int previous = -1;
		for(int lag : lags){
			if(previous < lag) previous = lag;
			else throw new IllegalArgumentException("Lags must be in ascending order");
		}
		
		if (oneRel) {
			for (int i = 0; i < maskInds.length; ++i) {
				maskInds[i]--;
			}
		}
		
		int nVox = volDims[0] * volDims[1] * volDims[2];
		double[][] lvMultiLags = new double[lags.length][nVox];
		int currentVoxel;
		int currIdx;
		
		for(int voxel = 0; voxel < singleLV.length; voxel+=winSize){
			currentVoxel = voxel / winSize;
			currIdx = maskInds[currentVoxel];
			
			if(currIdx < 0 || currIdx >= nVox){
				throw new IllegalArgumentException(
						"Invalid mask index element: " + 
						currIdx + ".  Largest valid value: " + (nVox - 1));
			}
			
			for(int i = 0; i < lags.length; i++){
				lvMultiLags[i][currIdx] =
					singleLV[voxel + lags[i]];
			}
		}
		
		writeVol4DNiftiDataset(volDims, voxSize, lags.length, filename,
				lvMultiLags);
	}

	/**
	 * @param volDims the voxel dimensions.
	 * @param voxSize voxel size
	 * @param units the number of cvs / lags depending on the datatype.
	 * @param filename the filename to write out this nifti information.
	 * @param vol2D either [lag#][lag] or [cv#][cv]
	 * @throws IOException 
	 * @throws FileNotFoundException
	 */
	private static void writeVol4DNiftiDataset(int[] volDims, double[] voxSize,
			int units, String filename, double[][] vol2D)
			throws IOException, FileNotFoundException {
		int datatype = 16;
		
		Nifti1Dataset niftiDS = new Nifti1Dataset();
		niftiDS.setHeaderFilename(filename);
		niftiDS.setDataFilename(filename);
		if (!MLFuncs.contains(DATA_TYPES, datatype)) {
			throw new IllegalArgumentException("Invalid datatype: " + datatype);
		}
		niftiDS.setDatatype((short)datatype);
		niftiDS.setVoxSize(voxSize);
		niftiDS.setDims((short)4, (short)volDims[0], (short)volDims[1], 
				(short)volDims[2],(short)units, (short)0, 
				(short)0, (short)0);
		
		niftiDS.writeHeader();
	 
		for(int unit = 0; unit < units; unit++ ){
			double[][][] vol3D = MLFuncs.reshapeZYX(vol2D[unit], 
									volDims[0], volDims[1], volDims[2]);
			niftiDS.writeVol(vol3D, (short)unit);
		}
	}
	
	/** Returns 3D vol dims of input 3D nifti vol
	 * 
	 * @param input 3D nifti vol filename
	 * @return volDims3D: 3-element int array {xdim, ydim, zdim}
	 */
	public static int[] getVolDims3D(String niftiFilename) throws IOException {
		int[] volDims3D = null;		
		Nifti1Dataset niftiDataset = new Nifti1Dataset(niftiFilename);
		// get dims
		try {
			niftiDataset.readHeader();
		}
		catch (FileNotFoundException e) {
			throw new IOException(e.getMessage());
		}
		volDims3D = new int[3];
		volDims3D[0] = niftiDataset.getXdim();
		volDims3D[1] = niftiDataset.getYdim();
		volDims3D[2] = niftiDataset.getZdim();
		
		return volDims3D;
	}

	/** Reads input nifti datafile into 2D array (returned).  
	 *  Rows of 2D array are timepoints (scans), and columns are voxels.
	 *  Hence each row contains a 1D representation of a 3D image.
	 *  Assumption: input data is arranged in [Z][Y][X] order, hence
	 *  each row of data in output 2D array is arranged in
	 *  [x + y*xdim + z*xdim*ydim] order.
	 *  Note that input datafile can be 3D or 4D.
	 * @param niftiFilename
	 * 		name of file containing data
	 * @param skipTmpts 
	 * 		indices of timepoints to skip (0-relative)
	 *      - Note that 'skipTmpts' can contain more points than the no. of tmpts in input file;
	 *        i.e., the tmpts in the current input file might be only a subset of a larger 
	 *        collection of skipped tmpts.  See 'firstTmpt'.  
	 * @param firstTmpt which tmpt (in skipTmpts) corresponds to first tmpt of curr. file.  E.g. if
	 *        the full set of tmpts is contained in 3 input data files (file1, file2, file3) and each 
	 *        file contains 10 tmpts, then to skip the first 3 tmpts in each file, skipTmpts 
	 *        = {0, 1, 2, 10, 11, 12, 20, 21, 22}.  When reading in file2 data, set firstTmpt = 10.
	 * @return 2D double array containing data (timepoints X voxels); if all timepoints in input
	 * 	       file are skipped, return null
	 * @throws IOException
	 */
	public static double[][] readNiftiData(String niftiFilename, int[] skipTmpts, int firstTmpt) 
		throws IOException {
		
		Nifti1Dataset niftiDataset = new Nifti1Dataset(niftiFilename);

		// get dims
		try {
			niftiDataset.readHeader();
		}
		catch (FileNotFoundException e) {
			throw new IOException(e.getMessage());
		}

		int xDim = niftiDataset.getXdim();
		int yDim = niftiDataset.getYdim();
		int zDim = niftiDataset.getZdim();
		int tDim = niftiDataset.getTdim();

		if (debug) {
			System.out.println("Xdim: " + xDim);
			System.out.println("Ydim: " + yDim);
			System.out.println("Zdim: " + zDim);
			System.out.println("Tdim: " + tDim);
		}

		if (tDim == 0) tDim = 1;

		// load data 
		long xyz = (long)xDim * (long)yDim * (long)zDim;
		//System.out.println("xDim * yDim * zDim: " + xyz);
		
		
		int[] currSkipT = getCurrSkipT(firstTmpt, tDim, skipTmpts);
		int nSkip = currSkipT.length;
		if (debug) {
			System.out.println("Skipping " + nSkip + " tmpts...");
		}
		double[][] dataArray2D = null;
		if ((tDim - nSkip) > 0) {
			// not all tmpts are skipped
			dataArray2D = new double[tDim - nSkip][(int)xyz];
			int i = 0;
			for (short t = 0; t < tDim; ++t) {
				if (MLFuncs.contains(currSkipT, t)) {
					if (debug) {
						System.out.println("Skipping tmpt " + t);
					}
				}
				else {
					if (debug) {
						System.out.println("Reading tmpt " + t);
					}
					dataArray2D[i] = niftiDataset.readDoubleVol1D(t);
					++i;
				}

//				double[] currVol = niftiDataset.readDoubleVolReturnArray(t);
				// switch from row (x)  to col (y) dominant
//				for (int z = 0; z < zDim; ++z) {
//				for (int y = 0; y < yDim; ++y) {
//				for (int x = 0; x < xDim; ++x) {
//				dataArray2D[t][y + (yDim * x) + (xDim * yDim * z)] = 
//				currVol[x + (xDim * y) + (xDim * yDim *z)];
//				}
//				}
//				}

			}
		}
		return dataArray2D;
	}
	
	/** Reads input nifti datafile into 2D byte array (returned) [doesn't check datatype]. 
	 *  Rows of 2D array are timepoints (scans), and columns are voxels.
	 *  Hence each row contains a 1D representation of a 3D image.
	 *  Assuming that input data is arranged in [x][y][z][t] order,
	 *  each row of data in output 2D array is arranged in
	 *  [x + y*xdim + z*xdim*ydim] order.
	 *  Note that input datafile can be 3D or 4D.
	 * @param niftiFilename
	 * 		name of file containing data
	 * @param skipTmpts 
	 * 		indices of timepoints to skip (0-relative)
	 *      - Note that 'skipTmpts' can contain more points than the no. of tmpts in input file;
	 *        i.e., the tmpts in the current input file might be only a subset of a larger 
	 *        collection of skipped tmpts.  See 'firstTmpt'.  
	 * @param firstTmpt which tmpt (in skipTmpts) corresponds to first tmpt of curr. file.  E.g. if
	 *        the full set of tmpts is contained in 3 input data files (file1, file2, file3) and each 
	 *        file contains 10 tmpts, then to skip the first 3 tmpts in each file, skipTmpts 
	 *        = {0, 1, 2, 10, 11, 12, 20, 21, 22}.  When reading in file2 data, set firstTmpt = 20.
	 * @return 2D byte array containing data (timepoints X voxels); if all timepoints in input
	 * 	       file are skipped, return null
	 * @throws IOException
	 */
	public static byte[][] readNiftiDataBytes(String niftiFilename, int[] skipTmpts, int firstTmpt) 
		throws IOException {
		
		Nifti1Dataset niftiDataset = new Nifti1Dataset(niftiFilename);

		// get dims
		try {
			niftiDataset.readHeader();
		}
		catch (FileNotFoundException e) {
			throw new IOException(e.getMessage());
		}

		int xDim = niftiDataset.getXdim();
		int yDim = niftiDataset.getYdim();
		int zDim = niftiDataset.getZdim();
		int tDim = niftiDataset.getTdim();

		if (debug) {
			System.out.println("Xdim: " + xDim);
			System.out.println("Ydim: " + yDim);
			System.out.println("Zdim: " + zDim);
			System.out.println("Tdim: " + tDim);
		}

		if (tDim == 0) tDim = 1;

		// load data 
		long xyz = (long)xDim * (long)yDim * (long)zDim;
		//System.out.println("xDim * yDim * zDim: " + xyz);
		
		int[] currSkipT = getCurrSkipT(firstTmpt, tDim, skipTmpts);
		int nSkip = currSkipT.length;
		if (debug) {
			System.out.println("Skipping " + nSkip + " tmpts...");
		}
		byte[][] dataArray2D = null;
		if ((tDim - nSkip) > 0) {
			// not all tmpts are skipped
			dataArray2D = new byte[tDim - nSkip][(int)xyz];
			int i = 0;
			for (short t = 0; t < tDim; ++t) {
				if (MLFuncs.contains(currSkipT, t)) {
					if (debug) {
						System.out.println("Skipping tmpt " + t);
					}
				}
				else {
					if (debug) {
						System.out.println("Reading tmpt " + t);
					}
					dataArray2D[i] = niftiDataset.readVolBlob(t);
					++i;
				}

//				double[] currVol = niftiDataset.readDoubleVolReturnArray(t);
				// switch from row (x)  to col (y) dominant
//				for (int z = 0; z < zDim; ++z) {
//				for (int y = 0; y < yDim; ++y) {
//				for (int x = 0; x < xDim; ++x) {
//				dataArray2D[t][y + (yDim * x) + (xDim * yDim * z)] = 
//				currVol[x + (xDim * y) + (xDim * yDim *z)];
//				}
//				}
//				}

			}
		}
		return dataArray2D;
	}
	
	
	/** Returns Matrix containing masked data.  Rows are timepoints (scans); columns
	 *  are voxels.  Voxels <= 0 are NOT excluded.
	 * @param maskIndices indices of voxels to be included (if null, calculate 
	 *  using input data) - NOT checked for validity!  
	 *  
	 *  @return Matrix of masked data
	 */
	protected static Matrix getMaskedDataMat(NpairsjSetupParams setupParams, String matlibType, int[] maskIndices) 
		throws MatrixException, IOException {
		
		int[] maskLocs = null;
		if (maskIndices == null) {
			// figure out mask indices from input data 
			double[] andMask = getANDMask(setupParams.getMaskFilenames());
			maskLocs = MLFuncs.findNonZero(andMask);
		}
		else maskLocs = maskIndices;
		
		int nMskVox = maskLocs.length;
		
		int nRowsIncl = setupParams.numVols;
		Matrix maskedData = new MatrixImpl(nRowsIncl, nMskVox, matlibType).getMatrix();
		
		// add data to Matrix one tmpt (row) at a time
		int row = 0; 
		int nFiles = setupParams.getDataFilenames().length;
		int firstTmpt = 0; // first tmpt in each file
		for (int f = 0; f < nFiles; ++f) {
			int currNTmpts = setupParams.getNTmptsPerFile()[f];
			int[] currSkipTmpts = getCurrSkipT(firstTmpt, currNTmpts, 
					setupParams.getSkipTmpts());
			firstTmpt += currNTmpts;	
			if (currSkipTmpts.length < currNTmpts) {
				// not all tmpts skipped in curr file
//				System.out.println("Reading " + f + "th file: " + setupParams.getDataFilenames()[f]);
				Nifti1Dataset data = new Nifti1Dataset(setupParams.getDataFilenames()[f]);
				try {
					data.readHeader();
				}
				catch (FileNotFoundException e) {
					throw new IOException(e.getMessage());
				}
				
				for (int t = 0; t < currNTmpts; ++t) {
					if (!MLFuncs.contains(currSkipTmpts, t)) {
						double[] currData = data.readDoubleVol1D((short)t);
						double[] currMaskedData = new double[nMskVox];
						for (int i = 0; i < nMskVox; ++i) {	
							currMaskedData[i] = currData[maskLocs[i]];
						}
						maskedData.setRowQuick(row, currMaskedData);
						++row;
					}
				}
			}
//			else {
//				System.out.println("Skipping " + f + "th file: " + setupParams.getDataFilenames()[f]);
//			}
		}		
		return maskedData;
	}
		
		
	/** Returns array containing indices of subset of skipTmpts within input range
	 *  of tDim tmpts beginning with firstTmpt: [firstTmpt, firstTmpt + tDim - 1]
	 * 
	 * @param firstTmpt
	 * @param tDim
	 * @param skipTmpts
	 * @return
	 */
	private static int[] getCurrSkipT(int firstTmpt, int tDim, int[] skipTmpts) {
		int[] uniqSkipT = new int[0];
		if (skipTmpts != null) {
			uniqSkipT = MLFuncs.sortAscending(MLFuncs.unique(skipTmpts));
		}
		if (debug) {
			System.out.println("First tmpt: " + firstTmpt);
		}
		// find indices of skipped tmpts greater than or equal to lower bound (firstTmpt)
		int[] skipIdxBoundBelow = MLFuncs.findGreaterThanOrEqualTo(uniqSkipT, firstTmpt);
//		 find indices of skipped tmpts less than or equal to upper bound (lastTmpt)
		int lastTmpt = firstTmpt + tDim - 1;
		int[] skipIdxBoundAbove = MLFuncs.findLessThanOrEqualTo(uniqSkipT, lastTmpt);
		// find intersection of bounded tmpts to get current subset of skipped tmpts
		// (i.e., tmpt indices)
		Vector<Integer> vCurrSkipT = new Vector<Integer>();
		for (int i = 0; i < skipIdxBoundBelow.length; ++i) {
			if (MLFuncs.contains(skipIdxBoundAbove, skipIdxBoundBelow[i])) {
				vCurrSkipT.add(uniqSkipT[skipIdxBoundBelow[i]]);
			}
		}
		int[] currSkipT = new int[vCurrSkipT.size()];
		for (int i = 0; i < currSkipT.length; ++i) {
			currSkipT[i] = vCurrSkipT.get(i) - firstTmpt;
		}
		return currSkipT;
	}

	/** Reads input nifti datafiles into 2D array (returned).  
	 *  Rows of 2D array are timepoints and columns are voxels.
	 *  Hence each row contains a 1D representation of a 3D image.
	 *  Assuming that input data is arranged in [x][y][z][t] order,
	 *  each row of data in output 2D array is arranged in
	 *  [x + y*xdim + z*xdim*ydim] order.
	 *  Note that input data can be 3D or 4D.
	 *  Data in output matrix is stacked in order of input niftiFilenames.
	 * @param niftiFilenames
	 * 			 String array of data filenames
	 * @return 2D double array containing data (timepoints X voxels) 
	 */
	public static double[][] readNiftiData(String[] niftiFilenames) throws IOException {

		double[][] dataArray2D = null;

		int numFiles = niftiFilenames.length;
		if (numFiles == 0) {
			throw new IOException("Error - no niftiFilenames in input arg");
		}

		int numTmpts = 0;
		int numVoxels = 0;
		Vector<double[]> data = new Vector<double[]>();

		for (int i = 0; i < numFiles; ++i) {
			if (debug) {
				System.out.println("Reading " + niftiFilenames[i] + "...");
			}
			int[] skipTmpts = null;
			double[][] currData2D = readNiftiData(niftiFilenames[i], skipTmpts, 0);
			numTmpts += currData2D.length;
			if (i == 0) {
				numVoxels = currData2D[0].length;
			}
			else if (numVoxels != currData2D[0].length) {
				throw new IOException("Error loading nifti data: " + niftiFilenames[i] + 
				" - no. of voxels in dataset does not match previously loaded datasets.");
			}
			for (int t = 0; t < currData2D.length; ++t) {
				data.add(currData2D[t]);
			}
		}
		if (debug) {
			System.out.println("Total number of tmpts: " + numTmpts);
			System.out.println("Number of voxels: " + numVoxels);
		}
		dataArray2D = new double[numTmpts][numVoxels];
		dataArray2D = data.toArray(dataArray2D);		

//		if (debug) {
//			System.out.println("Array to be returned by readData(String[]): ");
//			utils_tests.PCATest.printArray(dataArray2D);
//		}
		return dataArray2D;
	}
	
	
	/** Reads input nifti datafiles into 2D array (returned).
	 *  Input data files are masked with input masks, i.e.
	 *  voxels set to 0 in input masks are also set to 0
	 *  in corresponding data.
	 *  Masks are combined into ANDMask, which is then applied to all
	 *  input data 3D volumes. Data is read in order presented in
	 *  niftiFilenames.  
	 *  Rows of returned 2D array are timepoints, and columns are voxels.
	 *  Hence each row contains a 1D representation of a 3D image.
	 *  Assuming that input data is arranged in [x][y][z] order,
	 *  each row of data in output 2D array is arranged in
	 *  [x + y*xdim + z*xdim*ydim] order.
	 *  Note that input data can be 3D or 4D; masks must be 3D.
	 * @param niftiFilenames 
	 * 			String array of nifti data filenames
	 * @param maskFilenames
	 * 			String array of nifti mask filenames
	 * @return 2D double array containing data (timepoints X voxels) 
	 */
	public static double[][] readNiftiData(String[] niftiFilenames, String[] maskFilenames) 
			throws IOException {
		//TODO: save mask indices somewhere else, not in NiftiIO code.
		return readNiftiData(niftiFilenames, maskFilenames, null, null, null);
	}

	/** Reads input nifti datafiles into 2D array (returned).
	 *  Input data files are masked with input masks, i.e.
	 *  voxels set to 0 in input masks are also set to 0
	 *  in corresponding data.
	 *  Masks are combined into ANDMask, which is then applied to all
	 *  input data 3D volumes. Data is read in order presented in
	 *  niftiFilenames, skipping the timepoints at indices in 'skipTmpts'.  
	 *  Rows of returned 2D array are timepoints, and columns are voxels.
	 *  Hence each row contains a 1D representation of a 3D image.
	 *  Assuming that input data is arranged in [x][y][z] order,
	 *  each row of data in output 2D array is arranged in
	 *  [x + y*xdim + z*xdim*ydim] order.
	 *  Note that input data can be 3D or 4D; masks must be 3D.
	 * @param niftiFilenames 
	 * 			String array of nifti data filenames
	 * @param maskFilenames
	 * 			String array of nifti mask filenames
	 * @param saveMaskIndsFilename
	 *          String indicating filename under which to save indices of voxels included
	 *          in AND mask computed from input masks 
	 *          in an ASCII file (formatted to be read by idl read_matrix.pro)
	 *          - if null, then mask indices will not be saved
	 * @param skipTmpts 
	 * 			Timepoints to exclude when reading in data.  The indices 
	 *          run in order from 1st timepoint in niftiFilenames[0] to
	 *          last timepoint in niftiFilenames[n-1], where n = number of 
	 *          files in niftiFilenames.  
	 *          - if null, include all timepoints
	 * @param nTmptsPerFile how many timepoints exist in each input file (in order)
	 * @return 2D double array containing data (timepoints X voxels) 
	 */
	public static double[][] readNiftiData(String[] niftiFilenames, String[] maskFilenames, 
			String saveMaskIndsFilename, int[] skipTmpts, int[] nTmptsPerFile) throws IOException {
		int numDataFiles = niftiFilenames.length;
		int numMaskFiles = maskFilenames.length;

		if (numDataFiles == 0) {
			throw new IllegalArgumentException("Error loading nifti data-- input" +
					" niftiFilenames arg is empty");
		}
		if (numMaskFiles == 0) {
			throw new IllegalArgumentException("Error loading mask files-- input" +
					" maskFilenames arg is empty");
		}
		
		double[][] maskedData = null; 
	//	ArrayList<Double[]> mData;
		
		byte[] andMask = getANDMaskBytes(maskFilenames);

		// get size and indices of mask
		int nVox = andMask.length;
		Vector<Integer> vMaskIndices = new Vector<Integer>();
		
		for (int i = 0; i < nVox; ++i) {
			if (andMask[i] != 0) {
				vMaskIndices.add(i);
			}
		}
		int nMskVox = vMaskIndices.size();	
		
		if (saveMaskIndsFilename != null) {
			int[] maskIndices = new int[nMskVox];
	
			for (int j = 0; j < nMskVox; ++j) {
				maskIndices[j] = vMaskIndices.get(j).intValue();		
			}
			NpairsjIO.printToIDLFile(maskIndices, saveMaskIndsFilename);
		}

		//maskedData = new double[numDataFiles][nMskVox];
		int nScans = MLFuncs.sum(nTmptsPerFile); 
		int nSkipped = skipTmpts.length;
		
	//	mData = new ArrayList<Double[]>(nScans - nSkipped); 
		maskedData = new double[nScans - nSkipped][nMskVox];
		if (debug) {
			System.out.println("Size maskedData matrix: " + maskedData.length + " X " + 
					maskedData[0].length);
		}
		int currTmpt = 0;
		int currInclTmpt = 0;
		for (int i = 0; i < numDataFiles; ++i) {
			// load current dataset
			if (debug) {
				System.out.println("Reading datafile " + niftiFilenames[i] + "... ");		
				System.out.println("Current tmpt: " + currTmpt);
			}
			double[][] currData = readNiftiData(niftiFilenames[i], skipTmpts, currTmpt);
			if (currData == null) {
				// no tmpts included from current data file
				if (debug) {
					System.out.println("No tmpts incl from " + niftiFilenames[i]);
				}
				currTmpt += nTmptsPerFile[i];
				continue;
			}
			else {
				if (debug) {
					System.out.println("Incl tmpts from " + niftiFilenames[i]);
				}
				if (currData[0].length != nVox) {
					throw new IOException("Error-- data and mask images must have same "
							+ "dimensions");
				}
				
				int currNumScans = currData.length;
//				numScans += currNumScans - 1;
//				mData.ensureCapacity(numScans); // is this worth it?
				for (int k = 0; k < currNumScans; ++k) {
					int currRow = currInclTmpt + k;
//					Double[] currMaskedData = new Double[nMskVox];
					int maskedIndex = 0;
					for (int j = 0; j < nVox; ++j) {
						if (andMask[j] != 0) {
							maskedData[currRow][maskedIndex] = currData[k][j];
							++maskedIndex;
						}
					}
//					mData.add(currMaskedData);
				}
				currInclTmpt += currNumScans; // masked data row index incremented
			}
			currTmpt += nTmptsPerFile[i]; // total input data (incl and excl tmpts) index incremented
		}
//		maskedData = new double[nScans - nSkipped][nMskVox];
//		for (int n = 0; n < nScans - nSkipped; ++n) {
//			for (int v = 0; v < nMskVox; ++v) {
//				maskedData[n][v] = mData.get(n)[v];
//			}
//		}	
	
		if (maskedData == null) {
			throw new IOException("Could not create masked data");
		}
		return maskedData;
	}
	
	/** Returns data at input timepoint in input nifti file
	 * 
	 * @param niftiFilename
	 * 			name of file containing nifti data 
	 * @param tmpt
	 * 			timepoint to read
	 * @throws IllegalArgumentException
	 *          if timepoint is out of range
	 *          
	 * @return 1D double array containing input data at input timepoint
	 */
	public static double[] readNiftiData(String niftiFilename, int tmpt) throws IOException {
		
		Nifti1Dataset niftiDataset = new Nifti1Dataset(niftiFilename);
		// get dims
		try {
			niftiDataset.readHeader();
		}
		catch (FileNotFoundException e) {
			throw new IOException(e.getMessage());
		}
		
		int tDim = niftiDataset.getTdim();
		if (tDim == 0) tDim = 1;
			
		if (tmpt >= tDim) {
			throw new IllegalArgumentException("Timepoint " + tmpt + " is out of range - " +
					"input file " + niftiFilename + " has only " + tDim + " timepoints.");
		}		
		
		double[] data = niftiDataset.readDoubleVol1D((short)tmpt);
		return data;
	}
	
	/** Returns masked data at input timepoint in input nifti file
	 * 
	 * @param niftiFilename
	 * 			name of file containing nifti data 
	 * @param mask
	 * 			array containing 0's where data voxels are to be excluded
	 *  		- size of mask array must be same as 3d volume of data in input
	 *          nifti file
	 * @param tmpt
	 * 			timepoint to read
	 * @throws IllegalArgumentException
	 *          if timepoint is out of range or mask is incorrect size
	 *          
	 * @return 1D double array containing masked input data at input timepoint
	 */
public static double[] readNiftiData(String niftiFilename, double[] mask, int tmpt) 
		throws IOException {
		
		Nifti1Dataset niftiDataset = new Nifti1Dataset(niftiFilename);
		// get dims
		try {
			niftiDataset.readHeader();
		}
		catch (FileNotFoundException e) {
			throw new IOException(e.getMessage());
		}
		int xDim = niftiDataset.getXdim();
		int yDim = niftiDataset.getYdim();
		int zDim = niftiDataset.getZdim();
		int tDim = niftiDataset.getTdim();
		if (tDim == 0) tDim = 1;
			
		if (tmpt >= tDim) {
			throw new IllegalArgumentException("Timepoint " + tmpt + " is out of range - " +
					"input file " + niftiFilename + " has only " + tDim + " timepoints.");
		}
		long nVox = (long)xDim * (long)yDim * (long)zDim;		
		if (mask.length != nVox) {
			throw new IllegalArgumentException("Input mask and data are not the same size.");
		}
		
		double[] data = niftiDataset.readDoubleVol1D((short)tmpt);
		int nMskVox = MLFuncs.findNonZero(mask).length;
		double[] maskedData = new double[nMskVox];
		int maskedIndex = 0;
		for (int j = 0; j < nVox; ++j) {
			if (mask[j] != 0) {
				maskedData[maskedIndex] = data[j];
				++maskedIndex;
			}
		}
		
		return maskedData;
	}

/** Created to test Procrustes */ 
public static double[] readNiftiData(String niftiFilename, int tmpt, int[] maskCoords)
	throws IOException {
	Nifti1Dataset niftiDataset = new Nifti1Dataset(niftiFilename);
	try {
		niftiDataset.readHeader();
	}
	catch (FileNotFoundException e) {
		throw new IOException(e.getMessage());
	}
	double[] data = niftiDataset.readDoubleVol1D((short)tmpt);

	double[] maskedData = new double[maskCoords.length];
	int maskedIndex = 0;
	for (int j = 0; j < maskCoords.length; ++j) {
			maskedData[maskedIndex] = data[maskCoords[j] - 1];
			++maskedIndex;
	}
	return maskedData;
}

	
	/** Returns double 1D array containing AND mask corresponding to
	 * input setup parameters.  For each session file contained in 
	 * setup param file, data is masked in one of 3 ways:
	 * (i) using input mask file
	 * (ii) keeping all voxels (i.e. no data masked out)
	 * (iii) via thresholding technique a la PLS, i.e.:
	 * 		data from each input scan is masked by applying threshold
	 * 	    == T * scan max voxel value (where T is input threshold 
	 *      value between 0 and 1); intersection of all scan masks
	 *      is calculated to produce AND mask across all scans
	 * After a mask has been calculated for each session file via
	 * (i), (ii) or (iii), this method takes the intersection across
	 * session file masks to produce a grand AND mask.
	 *      
	 */
	public static double[] getANDMask(NpairsjSetupParams nsp) throws NpairsjException,
			IOException {
		String[] maskFilenames = nsp.getMaskFilenames();
		boolean[] useAllVox = nsp.inclAllVoxels();
		double[] threshVals = nsp.getMaskThreshVals();
		int nSess = maskFilenames.length;
		if (useAllVox.length != nSess
				||
			threshVals.length != nSess) {
			// Should never happen! If it does, there's a bug.
			throw new NpairsjException("Error: mask, incl. voxel and threshold info arrays " +
						"should all be the same length!");
		}
		
		double[] andMask = null;
		int maskSize = 0;
		for (int i = 0; i < nSess; ++i) {
			if (!useAllVox[i]) {
				if (maskFilenames[i] != null) {
					double[] currMask = getANDMask(new String[] {maskFilenames[i]});
					if (andMask == null) {
						andMask = currMask;
						maskSize = currMask.length;
					}
					else {
						if (currMask.length != maskSize) {
							throw new IllegalArgumentException("All data and masks must have the " +
									"same voxel dimensions.");					
						}
						// zero out voxels in andMask that are zero in current mask
						for (int j = 0; j < maskSize; ++j) {
							if (currMask[0] == 0) {
								andMask[0] = 0;
							}
						}
					}
						
				}
				else if (threshVals[i] != -1) {
					// no mask file supplied for curr sess file; calculate mask 
					// using threshold value
//					double[] currMask = getThreshMask(nsp, threshVals[i],
//							false);
					
				}
				else {
					// no mask or thresh info supplied even though useAllVox = false
					throw new NpairsjException("Error: data masking info for session file "
							+ "no. " + (nSess+1) + " is contradictory!");
				}
					
			}
			else {
				// using all voxels
			}
		}
			
		return null;
	}
	
	
	/** Returns mask generated by applying PLS-style thresholding to each
	 *  scan in current session and taking AND mask (intersection) of masks
	 *  generated for each scan.
	 *  PLS-style thresholding: multiply maximum voxel value in current scan
	 *  by input threshold to get lower bound on value of voxels to be included
	 *  in mask.  
	 * 
	 * @param NpairsjSetupParams nsp - .mat file containing npairs setup parameter info
	 * @param double thresh - value to multiply scan maximum by to get calculated
	 *                threshold for each scan
	 * @param boolean considerAllVoxels - if false, only voxels strictly greater than
	 *                calculated thresh will be included in mask
	 *                                  - if true, voxels equal to calculated thresh 
	 *                will also be included 
	 * @see #pls.sessionprofile.RunGenerateDatamat.findOnsetCoords(Matrix, double, boolean)
	 * @return
	 */
	private static double[] getThreshMask(NpairsjSetupParams nsp, int sessNum, double thresh, 
			boolean considerAllVoxels) {
		// which data files belong to this session?
		String[] sessDataFilenames = getSessDataFilenames(nsp, sessNum);
		
		
			
		
		
		
		return null;
	}

	
	/** Returns array (length total number of tmpts in given session file
	 * of data filenames by timepoint; if a given file is 4D
	 *  and contains multiple timepoints, said filename will be included once for each
	 *  timepoint it contains.
	 * @param nsp
	 * @param sessNum
	 * @return array of data filenames by timepoint
	 */
	private static String[] getSessDataFilenames(NpairsjSetupParams nsp,
			int sessNum) {
		// if given vol not included,
	    // corresponding element in paddedSessLabels will be '-1'
		int[] paddedSessLabels = getPaddedSessLabels(nsp);
		int[] currSessTmpts = MLFuncs.find(paddedSessLabels, sessNum); // 0-relative tmpts
		Set<String> currSessDataFilenames = new HashSet<String>();
		int nDataFiles = nsp.getDataFilenames().length;
		int currTmpt = 0;
		for (int i = 0; i < nDataFiles; ++i) {
			for (int j = 0; j < nsp.getNTmptsPerFile()[i]; ++j) {
				if (MLFuncs.contains(currSessTmpts, currTmpt)) {
					
				}
				++currTmpt;
			}
		}
		
		
		
		return null;
	}

	/** Returns session label array spanning all session
     file data volumes, not just the ones included
     in current analysis; if given vol not included,
     corresponding element in paddedSessLabels will be '-1'
     @param npairs setup parameters .mat file
     @return padded session label array
     */
	private static int[] getPaddedSessLabels(NpairsjSetupParams nsp) {
		int nVolTotal = nsp.numVols + nsp.getNSkipTmpts();
		int[] paddedSessLabels = new int[nVolTotal]; 
	    int skipIdx = 0;
	    int labIdx = 0;
		for (int i = 0; i < nVolTotal; ++i) {
	    	if (nsp.getSkipTmpts()[skipIdx] == i) {
	    		paddedSessLabels[i] = -1;
	    		++skipIdx;
	    	}
	    	else {
	    		paddedSessLabels[i] = nsp.getSessLabels()[labIdx];
	    		++labIdx;
	    	}	    	
	    }
		return paddedSessLabels;
	}

	/** Returns double 1D array containing AND mask created from input maskfiles
	 * (Maskfiles must be NIFTI or ANALYZE format)
	 *
	 * @param maskFileNames
	 * @return AND mask created from input maskfiles, as double array 
	 * 			(length num voxels in intersection of input masks)
	 */
	public static double[] getANDMask(String[] maskFilenames) throws IOException {
		int numMaskFiles = maskFilenames.length;
		double[][] andMask = null;
		int maskSize = 0;
		Vector<String> uniqMaskFiles = new Vector<String>();
		for (int i = 0; i < numMaskFiles; ++i) {
			if (debug) {
				System.out.println("Reading maskfile " + maskFilenames[i] + "...");
			}
			if (maskFilenames[i] != null && !uniqMaskFiles.contains(maskFilenames[i])) {
				uniqMaskFiles.add(maskFilenames[i]);
			}
		}
		for (String mf : uniqMaskFiles) {
			try {
				int[] skipTmpts = null;
				double[][] currMaskData = readNiftiData(mf, skipTmpts, 0);
				if (currMaskData == null || currMaskData.length != 1) {
					throw new IllegalArgumentException("Input masks must be 3D");
				}
				if (andMask == null) {
					andMask = currMaskData;
					maskSize = currMaskData[0].length;				
				}
				else {
					if (currMaskData[0].length != maskSize) {
						throw new IllegalArgumentException("All input masks must have the same dimensions");					
					}
				
					// zero out voxels in andMask that are zero in current mask
					for (int j = 0; j < maskSize; ++j) {
						if (currMaskData[0][j] == 0) {
							andMask[0][j] = 0;
						}
					}
				}
			}
			catch (IOException e) {
				throw new IOException(e.getMessage());
			}
			catch (NullPointerException npe) {
				throw new IOException("Unable to create AND Mask");
			}
		}
		if (andMask == null) {
			throw new IOException("Unable to create AND Mask");
		}
		
		return andMask[0];
	}
	
	
	/** Returns double 2D array containing AND mask created from input maskfiles
	 * (Maskfiles must be NIFTI or ANALYZE format)
	 *
	 * @param maskFileNames
	 * @return AND mask created from input maskfiles, as double 2D array 
	 * 			(dim 1 X num voxels in input masks)
	 */
	private static byte[] getANDMaskBytes(String[] maskFilenames) 
		throws IOException {
		if (debug) {
			System.out.println("Getting byte mask...");
		}
		int numMaskFiles = maskFilenames.length;
		byte[][] andMask = null;
		int maskSize = 0;
		Vector<String> uniqMaskFiles = new Vector<String>();
		for (int i = 0; i < numMaskFiles; ++i) {
			if (debug) {
				System.out.println("Reading maskfile " + maskFilenames[i] + "...");
			}
			if (!uniqMaskFiles.contains(maskFilenames[i])) {
				uniqMaskFiles.add(maskFilenames[i]);
			}
		}
		for (String mf : uniqMaskFiles) {
			try {
				byte[][] currMaskData = readNiftiDataBytes(mf, null, 0);
				if (currMaskData == null || currMaskData.length != 1) {
					throw new IllegalArgumentException("Input masks must be 3D");
				}
				if (andMask == null) {
					andMask = currMaskData;
					maskSize = currMaskData[0].length;				
				}
				else {
					if (currMaskData[0].length != maskSize) {
						throw new IllegalArgumentException("All input masks must have the same dimensions");					
					}
				
					// zero out voxels in andMask that are zero in current mask
					for (int j = 0; j < maskSize; ++j) {
						if (currMaskData[0][j] == 0) {
							andMask[0][j] = 0;
						}
					}
				}
			}
			catch (IOException e) {
				throw new IOException(e.getMessage());
			}
			catch (NullPointerException npe) {
				throw new IOException("Unable to create AND Mask");
			}
		}
		if (andMask == null) {
			throw new IOException("Unable to create AND Mask");
		}
		// strip extraneous 2nd dim
		return andMask[0];
		
	}
	
	
//	For testing (args[1] == nifti filename)
	
// NOTE: Careful!  x and y order is arranged so .img/hdr files look right when read in by 
// idl read_analyze; when saving as .nii, x and y dims were switched (so had [1][y][x] instead)
	public static void main2(String[] args) {
		Nifti1Dataset newNiftiDS;
		npairs.shared.matlib.ColtMatrix sampMat;
		npairs.shared.matlib.Matrix sampMat2;
		if (args[0].equals("mask")) {
			newNiftiDS = new Nifti1Dataset();
			
			newNiftiDS.setHeaderFilename(args[1] + ".nii");
		
			newNiftiDS.setDataFilename(args[1]);

			newNiftiDS.setDatatype((short)64);
			newNiftiDS.setDims((short)3, (short)4, (short)3, (short)1,
					(short)0, (short)0, (short)0, (short)0);
			sampMat = new npairs.shared.matlib.ColtMatrix(3,4);
			sampMat.setIdentity();
			System.out.println("Mask matrix: ");
			sampMat.print();
			sampMat2 = new npairs.shared.matlib.ColtMatrix(3,4);
			sampMat2.setIdentity();
			sampMat2.mult(2.0);
			System.out.println("Mask2 matrix: ");
			sampMat2.print();
		}

		else {
			newNiftiDS = new Nifti1Dataset();
			// set header filename first and include .nii ext to set dataset file type to .nii
			// (note: also need to write header before writing data when saving .nii file)
			// (note 2: doesn't matter whether .nii ext is included when setting data filename)
			newNiftiDS.setHeaderFilename(args[0] + ".nii");
			newNiftiDS.setDataFilename(args[0]);

			newNiftiDS.setDatatype((short)64);
			newNiftiDS.setDims((short)3, (short)4, (short)3, (short)2, 
					(short)0, (short)0, (short)0, (short)0);

			sampMat = new npairs.shared.matlib.ColtMatrix(3, 4);
			sampMat.setRandom();
			System.out.println("First random 2D colt matrix to be replicated in nifti dataset: ");
			sampMat.print();
			sampMat2 = sampMat.mult(2.0);
			System.out.println("Second 2D matrix to be replicated in nifti dataset: ");
			sampMat2.print();
		}
		// sampData3D is in [Z][Y][X] format, as per Nifti1Dataset.writeVol specs.
		double[][][] sampData3D = new double[2][3][4];
		sampData3D[0] = sampMat.toArray();
		sampData3D[1] = sampMat2.toArray();
		
		try {
			// remember: save header first when writing .nii files
			System.out.println("Writing header to disk... Filename: " + newNiftiDS.getHeaderFilename());
			newNiftiDS.writeHeader();
			System.out.println("Writing data to disk... Filename: " + newNiftiDS.getDataFilename());
			newNiftiDS.writeVol(sampData3D, (short)0);
			newNiftiDS.writeVol(sampData3D, (short)1);
			newNiftiDS.writeVol(sampData3D, (short)2);

		}
		catch (IOException e) {
			e.printStackTrace();
		}	
		
		// Now try loading info:
		try {
			int[] volDims = getVolDims3D(newNiftiDS.getDataFilename());
			System.out.println("Vol dims: ");
			NpairsjIO.print(volDims);
		}
		catch (IOException ioe) {
			ioe.printStackTrace();
		}
		
	}

	
	public static void main4(String[] args) {
		String datafile = "/home/anita/plsnpairs/spreng/data/RS4/001_bld008_rs4.nii";
		Nifti1Dataset dataNifti = new Nifti1Dataset(datafile);
		try {
			dataNifti.readHeader();
			dataNifti.printHeader();
		}
		catch (Exception e) {
			e.printStackTrace();
		}
	}
	/** Test masked volume write to disk */
	public static void main(String[] args) {
		
//		String dataFile = "/home/anita/plsnpairs/spreng/data/RS4/001_bld008_rs4.nii";
//		Nifti1Dataset dataNifti = new Nifti1Dataset(dataFile);
//		try {
//			dataNifti.readHeader();
//			dataNifti.printHeader();
//		}
//		catch (Exception e) {
//			e.printStackTrace();
//		}
		
		String resFile = "/home/anita/plsnpairs/spreng/results/10subj_30cls_allRuns_3splits/" +
			"10subj_30cls_allRuns_3splits_020pc_NPAIRSJresult.mat";
		// read in data and mask info
		ArrayList<String> fields = new ArrayList<String>();
		fields.add("st_coords");
		fields.add("npairs_result");
		fields.add("st_voxel_size");
		fields.add("st_origin");
		fields.add("st_dims");
		
		try {
			Map<String, MLArray> mResultInfo = new NewMatFileReader(
					resFile, new MatFileFilter(fields.toArray(new String[0]))).getContent();
			int[] st_coords = ((MLDouble) mResultInfo.get("st_coords")).getIntFirstRowOfArray();
			int[] volDims4D = ((MLDouble) mResultInfo.get("st_dims")).getIntFirstRowOfArray();
			int[] volDims = {volDims4D[0], volDims4D[1], volDims4D[3]};
			double[] voxSize = ((MLDouble) mResultInfo.get("st_voxel_size")).getFirstRowOfArray();
			voxSize[0] = -voxSize[0];
			int[] origin = ((MLDouble) mResultInfo.get("st_origin")).getIntFirstRowOfArray();
			
		    double[][] zs_eigim = ((MLDouble) ((MLStructure) mResultInfo.get("npairs_result")).
		    	getField("zscored_brainlv_avg")).getArray();
		    zs_eigim = MLFuncs.transpose(zs_eigim);
		    double[] zs_vol1D = zs_eigim[0];
		    System.out.println("Size st_coords: " + st_coords.length);
		    System.out.println("Size zs_vol1D: " + zs_vol1D.length);
		    
		    // save to disk
		    String saveName = "/home/anita/Desktop/10subj_30cls_allRuns_3splits_020pc.zs-eigim_voxqoffset.nii";
		    System.out.print("Saving " + saveName + "...");
		    writeVol(zs_vol1D, st_coords, true, volDims, voxSize, origin, saveName);
		    System.out.println("[DONE]");
		    
		 
		} catch (IOException ioe) {
			ioe.printStackTrace();
		}
	}
	
	
	/** Test volume write to disk */
	public static void main3(String[] args) {
		String dataFile = "/home/anita/Desktop/young_02_run_04_reg_align_sm_merge4d.nii";
		// read the file
		Nifti1Dataset nds = new Nifti1Dataset(dataFile);
		// get dims
		try {
			nds.readHeader();

			System.out.println("X dim: " + nds.getXdim());
			System.out.println("Y dim: " + nds.getYdim());
			System.out.println("Z dim: " + nds.getZdim());
			System.out.println("T dim: " + nds.getTdim());


			System.out.println("Reading 1st volume...");
			double[][][] vol3D0 = nds.readDoubleVol((short)0);
			System.out.println("Length vol3D0[0][0]: " + vol3D0[0][0].length);
			System.out.println("Length vol3D0[0]: " + vol3D0[0].length);
			System.out.println("Length vol3D0: " + vol3D0.length);
			
			System.out.println("Reading 1st volume as 1D array...");
			double[] vol1D0 = nds.readDoubleVol1D((short)0);
			System.out.println("Length vol1D0: " + vol1D0.length);
			
			String saveName3D = "/home/anita/Desktop/saved3DVolSep16VoxSz.nii";
			String saveName1D = "/home/anita/Desktop/saved1DVolSep16VoxSz.nii";
			double[] voxSize = {3.125, 3.125, 3.125};
			System.out.print("Saving " + saveName3D + "... ");
			writeVol(vol3D0, voxSize, saveName3D);
			System.out.println("[DONE]");
			System.out.print("Saving " + saveName1D + "... ");
//			writeVol(vol1D0, new int[] {nds.getXdim(), nds.getYdim(), nds.getZdim()}, voxSize, saveName1D);
			System.out.println("[DONE]");
			
//			double[][][] vol3Dredux = MLFuncs.reshapeZYX(vol1D0, nds.getXdim(), nds.getYdim(), 
//					nds.getZdim());
//			System.out.println("Length vol3Dredux[0][0]: " + vol3Dredux[0][0].length);
//			System.out.println("Length vol3Dredux[0]: " + vol3Dredux[0].length);
//			System.out.println("Length vol3Dredux: " + vol3Dredux.length);
//			
//			System.out.println("Are vol3Dredux and vol3D0 equal? " +
//					equal(vol3Dredux, vol3D0));
//			System.out.println("vol3D0[0][0][0] = " + vol3D0[0][0][0]);
//			System.out.println("vol3Dredux[0][0][0] = " + vol3Dredux[0][0][0]);
//			System.out.println("vol3D0[0][0][1] = " + vol3D0[0][0][1]);
//			System.out.println("vol3Dredux[0][0][1] = " + vol3Dredux[0][0][1]);
		} 
		catch (IOException ioe) {
			ioe.printStackTrace();
		}
		
	}
	
//	/** Required: all arrays in given dim have same length
//	 * 
//	 * @param a
//	 * @param b
//	 * @return
//	 */
//	private static boolean equal(double[][][] a, double[][][] b) {
//
//		if (a.length != b.length || a[0].length != b[0].length || 
//				a[0][0].length != b[0][0].length) {
//			return false;
//		}
//		for (int i = 0; i < a.length; ++i) {
//			for (int j = 0; j < a[0].length; ++j) {
//				if (!Arrays.equals(a[i][j], b[i][j])) {
//					return false;
//				}
//			}
//		}
//		return true;	
//	}
	 
}
