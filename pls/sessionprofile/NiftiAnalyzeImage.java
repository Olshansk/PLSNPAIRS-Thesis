package pls.sessionprofile;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.Vector;
import java.util.Arrays;
import javax.vecmath.Point3f;

import edu.washington.biostr.sig.brainj3d.volume.AnalyzeSPMDecoder;
import edu.washington.biostr.sig.brainj3d.volume.VolumeArray;
import edu.washington.biostr.sig.brainj3d.volume.VolumeArrayIterator;
import edu.washington.biostr.sig.brainj3d.volume.VolumeData;
import edu.washington.biostr.sig.nifti.*;
import pls.shared.MLFuncs;

import java.io.*;
import npairs.shared.matlib.ColtMatrix;

//import npairs.utils.NpairsIO;

public class NiftiAnalyzeImage {

	private VolumeData data = null;
    private NiftiFile file = null;
	private int maxX = 0;
	private int maxY = 0;
	private int maxZ = 0; 
	private int maxT = 0;
	
	private static int[] maskDims = null;
	
	public NiftiAnalyzeImage(String dataPath, String[] fileNames) throws Exception {
		
		URL[] urls = getHeaderURLS(dataPath, fileNames);
			AnalyzeSPMDecoder decoder = new AnalyzeSPMDecoder();
			data = (VolumeData)decoder.decode(dataPath, urls, null);	
			file = new NiftiFile(urls);
			
			maxX = data.getArray().getMaxX();
			maxY = data.getArray().getMaxY();
			maxZ = data.getArray().getMaxZ();
			maxT = data.getArray().getMaxTime();
	}
	
	public NiftiAnalyzeImage(String dataPath, String fileName) throws Exception {
		this(dataPath, new String[]{fileName});
	}
	
	public double[] getData(int[] sliceIndices) {
		VolumeArray array = data.getArray();
		//VolumeArray array = NiftiData;
		
		int numVoxels = (maxX * maxY * sliceIndices.length);
		double[] ret = new double[numVoxels];
		int colCount = 0, currSlice = -1;
	    for(VolumeArrayIterator it = array.iterator(); it.hasNext(); colCount++) {
	    	if(colCount % (maxX * maxY) == 0) {
		    	currSlice++;
		    }
		    if(MLFuncs.contains(sliceIndices, currSlice)) {
				ret[colCount] = it.nextDouble();
		    } else {
		    	it.next();
		    }
		}
	    return ret;
	}
	
	public double[] getData() {
		int[] sliceIndices = new int[maxZ];
		for (int i = 0; i < maxZ; ++i) {
			sliceIndices[i] = i;
		}
		return getData(sliceIndices);
	}
	
	/**
	 * Handles getting the header and image filenames from both nii and analyze
	 * @param dataPath path to the data
	 * @param fileNames file names
	 * @return array of URLS for the header and image files
	 * @throws Exception malformed URL
	 */
	public static URL[] getHeaderURLS(String dataPath, String[] fileNames) throws Exception {
		Vector<URL> runImages = new Vector<URL>();
		for(String fileName : fileNames) {
			File file = new File(dataPath, fileName);
			runImages.add(file.toURL());
			if(fileName.toLowerCase().endsWith("img")) {
				String headerName = fileName.split("img")[0] + "hdr";
				File headerFile = new File(dataPath, headerName);
				runImages.add(headerFile.toURL());
			} else if (fileName.toLowerCase().endsWith("hdr")) {
				String imageName = fileName.split("hdr")[0] + "img";
				File imageFile = new File(dataPath, imageName);
				runImages.add(imageFile.toURL());
			}
		}
		Object[] items = runImages.toArray();
		URL[] urls = new URL[items.length];
		System.arraycopy(items, 0, urls, 0, items.length);
		return urls;
	}
	
	public int[] getDimensions() {
		return new int[]{maxX, maxY, maxZ};
	}
	
	public int[] get4DDimensions() {
		return new int[] {maxX, maxY, maxZ, maxT};		
	}
	
	public float[] getVoxelSize() {
	      Point3f voxel = new Point3f();
	      Point3f voxel1 = new Point3f(1f, 1f, 1f);
	      Point3f voxel2 = new Point3f(0f, 0f, 0f);
	      data.getIndex2Space().transform(voxel1);
	      data.getIndex2Space().transform(voxel2);
	      voxel.sub(voxel1, voxel2);
	      voxel.absolute();
	      return new float[]{voxel.x, voxel.y, voxel.z};
	}
	
	public short[] getOrigin() {
		return file.getHeader().getOrigin();
		/*float[] srowX = file.getHeader().getSrowX();
		float[] srowY = file.getHeader().getSrowY();
		float[] srowZ = file.getHeader().getSrowZ();
		float x = -(srowX[3] / srowX[0]) + 1;
		float y = -(srowY[3] / srowY[1]) + 1;
		float z = -(srowZ[3] / srowZ[2]) + 1;
		return new float[]{x, y, z};*/
	}
	
	/** Returns new 2D double array containing masked data.
	 *  Each row consists of one (3D) data volume; masked-out voxels are excluded,
	 *  so row length will be <= original data volume size.
	 *  Mask volumes are combined to create AND mask which is then applied
	 *  to all input data volumes.
	 * @param dataFileNames
	 * @param maskFileNames
	 * @return double[][] array containing only non-masked voxels
	 * @throws Exception
	 */
	public static double[][] getMaskedData(String[] dataFileNames, String[] maskFileNames) 
			throws Exception {
		
		if (dataFileNames.length != maskFileNames.length) {
			throw new IllegalArgumentException(
					"Number of data volumes and mask volumes must be the same");
		}
		
		double[][] maskedData = null;
		try {
			double[] andMask = getANDMask(maskFileNames);
			int maskSize = 0;
			for (double vox : andMask) {
				if (vox != 0) {
					++maskSize;
				}
			}
			maskedData = new double[dataFileNames.length][maskSize];
			
			for (int i = 0; i < dataFileNames.length; ++i) {
				// parse current dataFilename into pathname and filename
				String currDataFile = dataFileNames[i];
				int dataPathNameEnd = currDataFile.lastIndexOf(System.getProperty(
						"file.separator"));
				String dataFileName = currDataFile.substring(dataPathNameEnd + 1).trim();
				System.out.println("Image filename: " + dataFileName);
				String dataPath = currDataFile.substring(0, dataPathNameEnd);
				System.out.println("Image path: " + dataPath);
				try {
					// load current data volume
					NiftiAnalyzeImage dataImage = new NiftiAnalyzeImage(dataPath, dataFileName);
				
					if (!Arrays.equals(maskDims, dataImage.getDimensions())) {
						throw new IllegalArgumentException(
								"All data volumes and mask volumes must have the "
							  + "same dimensions");
					}
					
					// mask data 
					double[] currData = dataImage.getData();
					int currNonZeroVox = 0;
					for (int j = 0; j < currData.length; ++j) {
						if (andMask[j] != 0) {
							maskedData[i][currNonZeroVox] = currData[j];
							++currNonZeroVox;
						}
					}					
				}
				catch (IllegalArgumentException e) {
					throw new IllegalArgumentException(e);
				}
				catch (Exception e) {
					throw new Exception(e);
				}
			}
		}
		catch (IllegalArgumentException e) {
			throw new IllegalArgumentException(e);
		}
		catch (Exception e) {
			throw new Exception(e);
		}
		
		return maskedData;
	}
	
	/** Returns double array containing AND mask created from input maskfiles
	 * (Maskfiles must be NIFTI or ANALYZE format)
	 *
	 * @param maskFileNames
	 * @return AND mask created from input maskfiles, as double array
	 */
	public static double[] getANDMask(String[] maskFileNames) throws Exception {
		double[] andMask = null;
		
		for (int i = 0; i < maskFileNames.length; ++i) {
			// parse current maskFilename into pathname and filename			
			String currMaskFile = maskFileNames[i];
			int maskPathNameEnd = currMaskFile.lastIndexOf(System.getProperty(
					"file.separator"));
			System.out.println("Mask path name end: " + maskPathNameEnd);
			String maskFileName = currMaskFile.substring(maskPathNameEnd + 1);
			System.out.println("Mask filename: " + maskFileName);
			String maskPath = currMaskFile.substring(0, maskPathNameEnd);
			System.out.println("Mask path: " + maskPath);
			// load current mask volume			
			try {
				NiftiAnalyzeImage maskImage = new NiftiAnalyzeImage(maskPath, maskFileName);
				if (i == 0) {
					maskDims = maskImage.getDimensions();
					andMask = maskImage.getData();
				}
				else if (!Arrays.equals(maskDims, maskImage.getDimensions())) {
					throw new IllegalArgumentException("All input mask volumes must have the same dimensions");					
				}
				else {
					// zero out voxels in andMask that are also zero in current mask:
					double[] currMask = maskImage.getData();
					for (int j = 0; j < currMask.length; ++j) {
						if (currMask[j] == 0) {
							andMask[j] = 0;
						}
					}
				}
			}
			catch(Exception e){
				throw new Exception(e);
			}
		}	
		
		if (andMask == null) {
			throw new Exception("Unable to create AND mask");
		}
		
		return andMask;
	}
	
	// for testing
	public static void main (String[] args) {
		String imageFilenameList = "/haier/anita/volList.txt";
		String maskFilenameList = "/haier/anita/maskVol.txt";
		
		//read in vol filenames
		Vector<String> imageFilenames = new Vector<String>();
		String volDir = "/haier/data/grady/snodgrass/fMRI/normalization/data/vol3D_analyze/" +
		"young_02_run_04_ras_ortho/";
		try {
			BufferedReader br = new BufferedReader(new FileReader(imageFilenameList));
			String nextFilename = br.readLine();
			while(nextFilename != null) {
				imageFilenames.add(volDir + nextFilename);
				nextFilename = br.readLine();
			}
		}
		catch (IOException e) {
			e.printStackTrace();
		}
		//read in mask filenames
		Vector<String> maskFilenames = new Vector<String>();
		String maskDir = "/haier/data/grady/snodgrass/fMRI/normalization/data/masks/";
		try {
			BufferedReader br = new BufferedReader(new FileReader(maskFilenameList));
			String nextFilename = br.readLine();
			while(nextFilename != null) {
				maskFilenames.add(maskDir + nextFilename);
				nextFilename = br.readLine();
			}
		}
		catch (IOException e) {
			e.printStackTrace();
		}
		
		String[] imageFilenamesS = new String[imageFilenames.size()];
		imageFilenamesS = imageFilenames.toArray(imageFilenamesS);
		
		String[] maskFilenamesS = new String[maskFilenames.size()];
		maskFilenamesS = maskFilenames.toArray(maskFilenamesS);
		
		//print filename lists
		System.out.println("IMAGE FILENAMES: ");
		for (String imageFname : imageFilenamesS) {
			System.out.println(imageFname);
		}
		System.out.println(); 
		System.out.println("MASK FILENAMES:");
		for (String maskFname : maskFilenamesS) {
			System.out.println(maskFname);
		}
		
		//load masked data
		try {
			double[][] maskedData = getMaskedData(imageFilenamesS, maskFilenamesS);
			ColtMatrix maskedDataMat = new ColtMatrix(maskedData);
			maskedDataMat.printToFile("/haier/anita/maskedNiftiAnalyzeImageDataMat.idl", "IDL");
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		System.out.println("Finishing up...");
	}
	
	public static void main1(String[] args) {
		String maskFilename = "young_02_mean_align_mask.img";
		String maskPath = "/haier/data/grady/snodgrass/fMRI/normalization/data/masks";
		String dataFilename = "young_02_run_04_ras_ortho_0005.img";
		String dataPath = "/haier/data/grady/snodgrass/fMRI/normalization/data/vol3D_analyze/young_02_run_04_ras_ortho";
		
		System.out.println("Loading mask...");
		try {
			NiftiAnalyzeImage maskNAI = new NiftiAnalyzeImage(maskPath, maskFilename);
			System.out.println("Getting mask 1D data array...");
			double[] maskArray = maskNAI.getData();
			System.out.println("Size of mask array: " + maskArray.length);
		} catch (Exception e) {
			e.printStackTrace();
		}
		System.out.println("Loading image...");
		try {
			NiftiAnalyzeImage imageNAI = new NiftiAnalyzeImage(dataPath, dataFilename);
			System.out.println("Getting image 1D array...");
			double[] imageArray = imageNAI.getData();
			System.out.println("Size of image array: " + imageArray.length);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		
	}
}


