package pls.test;

import java.io.File;

import extern.ArrayFuncs;

import npairs.io.NpairsjIO;
import extern.niftijlib.Nifti1Dataset;
import pls.sessionprofile.NiftiAnalyzeImage;

public class NiftiDataLoadTest {
	
	public NiftiDataLoadTest(String dataDir, String dataFilename) throws Exception {
		
		NiftiAnalyzeImage dataVol = new NiftiAnalyzeImage(dataDir, dataFilename);
		short[] origin = dataVol.getOrigin();
		System.out.println("NiftiAnalyzeImage origin: ");
		NpairsjIO.print(origin);
		int[] dims = dataVol.getDimensions();
		System.out.println("NiftiAnalyzeImage dims: ");
		NpairsjIO.print(dims);
		float[] voxelSize = dataVol.getVoxelSize();
		System.out.println("NiftiAnalyzeImage voxel size: ");
		NpairsjIO.print(voxelSize);
		
		String dataPath = (new File(dataDir, dataFilename).getPath());
		System.out.println("data file: " + dataPath);
		Nifti1Dataset nDS = new Nifti1Dataset(dataPath);
		nDS.readHeader();
		short[] dims2 = new short[] {nDS.getXdim(), nDS.getYdim(), nDS.getZdim()};
		System.out.println("Nifti1Dataset dims: ");
		NpairsjIO.print(dims2);
		float[] voxSize2 = nDS.pixdim;
		System.out.println("Nifti1Dataset pixdim (== voxel size? ): ");
		NpairsjIO.print(voxSize2);
		float[] voxSize3d = new float[] {voxSize2[1], voxSize2[2], voxSize2[3]};
		float[] qoffset = nDS.qoffset;
		System.out.println("Nifti1Dataset qoffset (== origin?):");
		NpairsjIO.print(qoffset);
		// take qoffset and calculate origin in voxels
		//  - for each dim, origin = -(qoffset) / (voxsize)
		int[] originAsVox = new int[3];
		for (int i = 0; i < 3; ++i) {
			originAsVox[i] = (int) (-qoffset[i] / voxSize3d[i]) + 1; // add one because
			                                                         // voxels are 1-rel.
		}
//		int[] originAsInts = (int[])ArrayFuncs.convertArray(qoffset, int.class);
		System.out.println("Nifti1Dataset origin as vox: ");
		NpairsjIO.print(originAsVox);	
		
	}
	
	public static void main (String[] args) {
		String dataDir = args[0];
		String dataFilename = args[1];
		try {
		NiftiDataLoadTest test = new NiftiDataLoadTest(dataDir, dataFilename);
		} catch (Exception e) { e.printStackTrace(); }
	}
		

}
