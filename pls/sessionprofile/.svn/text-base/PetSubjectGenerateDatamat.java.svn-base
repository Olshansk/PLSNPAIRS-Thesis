package pls.sessionprofile;

import java.util.Vector;
import java.io.File;
import java.net.URL;

import javax.swing.JOptionPane;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLStructure;

import pls.chrome.shared.ProgressDialogWatcher;
import pls.shared.MLFuncs;
import pls.sessionprofile.PetSubjectInformation;

import Jama.Matrix;

import edu.washington.biostr.sig.nifti.NiftiFile;
import extern.ArrayFuncs;

public class PetSubjectGenerateDatamat extends ProgressDialogWatcher {
	
	private String sessionFile = null;
	private boolean useBrainMask = false;
	private String brainMaskFile = null;
	private double coordThresh = 0;
	private boolean normalizeMeanVolume = false;
	private boolean considerAllVoxels = false;
	private boolean mergedConditions;
	private String [] mergedconditionList;
	/*
	private boolean isBlockedFmri = false;
	private int[] ignoreRuns = null;
	private int[] ignoreSlices = null;
	
	private int numSkippedScans = 0;
	private int windowSize = 0;
	private boolean mergeAcrossRunsFlag = false;
	private boolean normalizeSignalMean = false;
	private boolean singleSubject = false;
	
	
    */
	
	// {Condition Name, Reference Scan Onset, Number of Reference Scans}
	private Vector<String[]> conditionInfo = null;
	private Vector<PetSubjectInformation> subjectInfo = null;
	private String datamatPrefix = null;
	
	public PetSubjectGenerateDatamat(String sessionFile, boolean useBrainMask, String brainMaskFile, double coordThresh, boolean normalizeMeanVolume,Vector<String[]> conditionInfo, Vector<PetSubjectInformation> subjectInfo, String datamatPrefix,  boolean mergedConditions, String [] mergedconditionList) {
		this.datamatPrefix = datamatPrefix;
		this.sessionFile = sessionFile;
		this.useBrainMask = useBrainMask;
		this.brainMaskFile = brainMaskFile;
		this.coordThresh = coordThresh;
		this.conditionInfo = conditionInfo;
		this.subjectInfo = subjectInfo;
		this.normalizeMeanVolume = normalizeMeanVolume;
		this.mergedConditions= mergedConditions;
		this.mergedconditionList = mergedconditionList;
		
		if(this.coordThresh==0)
			considerAllVoxels=true;
		
		/*		
		this.isBlockedFmri = isBlockedFmri;
		this.ignoreRuns = ignoreRuns;		
		this.ignoreSlices = ignoreSlices;
		
		this.numSkippedScans = numSkippedScans;
		if(isBlockedFmri) {
			this.windowSize = 1;
		} else {
			this.windowSize = windowSize;
		}
		this.mergeAcrossRunsFlag = mergeAcrossRunsFlag;
		this.normalizeSignalMean = normalizeSignalMean;		
		this.singleSubject = singleSubject;
		this.considerAllVoxels = considerAllVoxels;
		*/
	}
	
	public PetSubjectGenerateDatamat(String sessionFile2, Vector<String[]> conditionInfo2, Vector<PetSubjectInformation> subjectInfo, String datamatPrefix2) {
		
	//	this.numSkippedScans = numSkippedScans;
		this.conditionInfo = conditionInfo;
		this.subjectInfo = subjectInfo;
		this.datamatPrefix = datamatPrefix;
	}

	/**
	 * Get the maximum number of onsets within a run
	 * @param run a run
	 * @return the maximum number of onsets within a run
	 */
	private int findMaxOnsets(PetRunInformation run) {
		int maxOnsets = 0;
		for(int i = 0; i < run.onsets.length; i++) {
			String[] onsets = run.onsets[i].split(" ");
			if(onsets.length > maxOnsets) {
				maxOnsets = onsets.length;
			}
		}
		return maxOnsets;
	}
	
	public void doTask() throws Exception {
		
		int numConditions = conditionInfo.size();
		int numSubject = subjectInfo.size();
		
		int[] dims = null;
		double[] voxelSize = null;
		double[] origin = null;
		
		try {
			// Get image files path/file name <for read only one image file>
			String dataPath = subjectInfo.get(0).dataDirectory;
			String fileName = subjectInfo.get(0).subjectFiles[0];	
			NiftiAnalyzeImage image = new NiftiAnalyzeImage(dataPath, fileName);
			int[] dimsTemp = image.getDimensions();
			dims = new int[]{dimsTemp[0], dimsTemp[1], 1, dimsTemp[2]};			
			voxelSize = (double[])ArrayFuncs.convertArray(image.getVoxelSize(), double.class);
			origin = (double[])ArrayFuncs.convertArray(image.getOrigin(), double.class);
		} catch(Exception ex) {
			throw new Exception("Something bad happened when reading nifti", ex);
		}
		
		int numVoxels = MLFuncs.product(dims);
		int[]coords = null;
		int [] m = null;
		//int[] sliceIndeces = MLFuncs.removeAll(MLFuncs.range(0, (int)dims[3] - 1), MLFuncs.subtract(ignoreSlices, 1));		
		int[] sliceIndeces = MLFuncs.range(0, (int)dims[3] - 1);
	/*	
 	for(int i=0; i<origin.length;i++)
			System.out.println("origin "+i+origin[i]);
			*/
		if(useBrainMask) {
			NiftiAnalyzeImage maskImage = null;
			try {
				File file = new File(brainMaskFile);
				maskImage = new NiftiAnalyzeImage(file.getParent(), file.getName());
				
				int[] dimsTemp = maskImage.getDimensions();
				int[] maskDims = new int[]{dimsTemp[0], dimsTemp[1], 1, dimsTemp[2]};
				
				if(!MLFuncs.isEqual(maskDims, dims)) {
					throw new Exception("Dimensions of the data do not match that of the brain mask!");
				}
				progress.postMessage("Same Dimensions with brain mask file");
								
				double[] brainMask = maskImage.getData(sliceIndeces);
				
				coords = MLFuncs.findGreaterThan(brainMask, 0);

				
			} catch(Exception ex) {
				throw new Exception("Unable to read mask file " + brainMaskFile, ex);
			}
		} else {
			coords = MLFuncs.zeros(numVoxels);
		}
		progress.startTask("Initialize Creating Datamat");
		
		//1st section of progress bar
		int section1 = numSubject * numConditions/(numSubject*numConditions+10);
		//factor for the second section
		double factor = 1/(numSubject*numConditions+10);
		
		//progress.updateStatus("****",(int)(0.5*factor));
		
		Matrix temp = null;
		Matrix datamat = null;
		Matrix tdatamat = null;
		Matrix tdataset = null;
		int numFiles;
		
		//make tdatamat, which includes non brain voxels
		
		for(int i=0;i<numConditions; i++)
		{
			temp=null;
			
			for(int j=0;j<numSubject;j++ ){

				progress.startTask("Loading condition " + (i + 1) + " of subject " + (j + 1));
				PetSubjectInformation thisSubectInfo = subjectInfo.get(j);
				String dataPath = thisSubectInfo.dataDirectory;
				String[] fileList = thisSubectInfo.subjectFiles;

				try {
					URL[] url = NiftiAnalyzeImage.getHeaderURLS(dataPath, new String[]{fileList[i]});
					NiftiFile file = new NiftiFile(url);
					numFiles = file.getHeader().getDim()[4]; // Time frame of a dataset
					NiftiAnalyzeImage image = new NiftiAnalyzeImage(dataPath, fileList[i]);
					
					tdataset = new Matrix(image.getData(sliceIndeces), 1);
					
					temp = MLFuncs.append(temp, tdataset);
					if(!useBrainMask) {
						coords = MLFuncs.sum(coords, findNonBrainCoords(tdataset, coordThresh, considerAllVoxels));
					}
						
				} catch(Exception ex) {
					throw new Exception("Unable to read file " + fileList[i] + " from path " + dataPath, ex);
				}
				progress.endTask();
			}
			tdatamat = MLFuncs.append(tdatamat, temp);
			
			progress.postMessage(" "+(i-1*numSubject)*factor);
		}
						
		progress.postMessage("Selecting only the brain voxls");

		//determine the coords of the brain region
		if(!useBrainMask) { // coords from thres by find non brain coords
			coords = MLFuncs.find(coords,0);
		}
		int dr = tdatamat.getRowDimension();
		int dc = tdatamat.getColumnDimension();
		
		factor = 0.1/dr;
		for(int i=0; i<dr; i++)
			progress.postMessage(" "+ section1+(i*factor)*(1-section1));
		
		
		//remap data to eliminate non-brain voxels
		datamat = MLFuncs.getColumns(tdatamat, coords);

		dr = datamat.getRowDimension(); // update dr/dc
		dc = datamat.getColumnDimension();
		
		Matrix gmean = null;
		
		if(normalizeMeanVolume) {
			//perform whole-brain ratio adjustment
		
			gmean = MLFuncs.rowMean(datamat);//grand mean for each image

			progress.postMessage("Normalizing datamat "+ section1+0.2*(1-section1));
			
			progress.postMessage("Normalizing datamat with its volume mean...");
			
			factor = 0.8/dc;
			double checkpoint = dc/10;
			int check = (int)checkpoint;
			int percentage = 10;
			
			for(int i=0; i<dc; i++)
			{	
				//normalized on the meanof each img
				
				Matrix den = datamat.getMatrix(0,dr-1,i,i);
				den = den.transpose().arrayRightDivide(gmean);
				den = den.transpose(); 
				
				datamat.setMatrix(0,dr-1,i,i,den);
				if(i==check)
				{
					progress.postMessage(""+ section1+(0.2+i*factor)*(1-section1));
					progress.postMessage(""+percentage);
					check = check + (int)checkpoint;
					percentage +=10;					
				}
			}
		}
		
		progress.endTask();
		/*
		if(mergedConditions){
			
		}
		*/
		progress.postMessage("Postprocessing to shape the datamat");

		String extension = "_PETdatamat.mat";
		String fileName = datamatPrefix + extension;

		progress.startTask("Saving to file " + fileName);

		try {	
			new PetDatamatSaver(datamat, coords, dims, voxelSize, origin, sessionFile, normalizeMeanVolume, brainMaskFile, coordThresh, considerAllVoxels, normalizeMeanVolume, fileName);																						
		} catch(Exception ex) {
			throw new Exception("Unable to write to file " + fileName, ex);
		}
		progress.endTask();
		
//		progress.updateStatus("Completed", 1);
		progress.complete();
	}
	
	private int[] findNonBrainCoords(Matrix dataset, double coordThresh, boolean considerAllVoxels) {
		int numScans = dataset.getRowDimension();
		int numVoxels = dataset.getColumnDimension();
		
		int[] nonBrainCoords = new int[numVoxels];
		
		for(int i = 0; i < numScans; i++) {
			double scanThreshold = MLFuncs.max(dataset.getArray()[i]) * coordThresh;
			
			for(int j = 0; j < numVoxels; j++) {
				if(considerAllVoxels && dataset.getArray()[i][j] < scanThreshold) {
					nonBrainCoords[j] = 1;
				} else if(!considerAllVoxels && dataset.getArray()[i][j] <= scanThreshold) {
					nonBrainCoords[j] = 1;
				}
			}
		}
		return nonBrainCoords;
	}
}
