package pls.sessionprofile;

import java.util.Vector;
import java.io.File;
import java.net.URL;

import npairs.io.NiftiIO;

import pls.chrome.shared.ProgressDialogWatcher;
import pls.shared.BfMRIDatamatFileFilter;
import pls.shared.MLFuncs;
import pls.shared.fMRIDatamatFileFilter;
import pls.sessionprofile.RunInformation;

import Jama.Matrix;

import edu.washington.biostr.sig.nifti.NiftiFile;
import extern.ArrayFuncs;
import extern.niftijlib.Nifti1Dataset;

public class RunGenerateDatamat extends ProgressDialogWatcher {
	
	private static final File File  = null;
	private boolean isBlockedFmri = false;
	private int[] ignoreRuns = null;
	private String sessionFile = null;
	private boolean useBrainMask = false;
	private String brainMaskFile = null;
	private double coordThresh = 0;
	private int[] ignoreSlices = null;
	private boolean normalizeMeanVolume = false;
	private int numSkippedScans = 0;
	private int windowSize = 0;
	private boolean mergeAcrossRunsFlag = false;
	private boolean normalizeSignalMean = false;
	private boolean considerAllVoxels = false;
	private boolean singleSubject = false;
	// {Condition Name, Reference Scan Onset, Number of Reference Scans}
	private Vector<String[]> conditionInfo = null;
	private Vector<RunInformation> runInfo = null;
	private String datamatPrefix = null;
	
	public RunGenerateDatamat(boolean isBlockedFmri, int[] ignoreRuns, String sessionFile, boolean useBrainMask, 
			String brainMaskFile, double coordThresh, int[] ignoreSlices, boolean normalizeMeanVolume, 
			int numSkippedScans, int windowSize, boolean mergeAcrossRunsFlag, boolean normalizeSignalMean, 
			boolean considerAllVoxels, boolean singleSubject, Vector<String[]> conditionInfo, 
			Vector<RunInformation> runInfo, String datamatPrefix) {
		this.isBlockedFmri = isBlockedFmri;
		this.ignoreRuns = ignoreRuns;
		this.sessionFile = sessionFile;
		this.useBrainMask = useBrainMask;
		this.brainMaskFile = brainMaskFile;
		this.coordThresh = coordThresh;
		this.ignoreSlices = ignoreSlices;
		this.normalizeMeanVolume = normalizeMeanVolume;
		this.numSkippedScans = numSkippedScans;
		if(isBlockedFmri) {
			this.windowSize = 1;
		} else {
			this.windowSize = windowSize;
		}
		this.mergeAcrossRunsFlag = mergeAcrossRunsFlag;
		this.normalizeSignalMean = normalizeSignalMean;
		this.considerAllVoxels = considerAllVoxels;
		this.singleSubject = singleSubject;
		this.conditionInfo = conditionInfo;
		this.runInfo = runInfo;
		this.datamatPrefix = datamatPrefix;
	}
	
	/**
	 * Get the maximum number of onsets within a run
	 * @param run a run
	 * @return the maximum number of onsets within a run
	 */
	private int findMaxOnsets(RunInformation run) {
		int maxOnsets = 0;
		for(int i = 0; i < run.onsets.size(); i++) {
			String[] onsets = run.onsets.get(i).split(" ");
			if(onsets.length > maxOnsets) {
				maxOnsets = onsets.length;
			}
		}
		return maxOnsets;
	}
	
	public void doTask() throws Exception {
		int maxOnsets = 0;
		int numConditions = conditionInfo.size();
		int numRuns = runInfo.size();
		
		// Get the maximum number of onsets
		for(int i = 0; i < numRuns; i++) {
			int temp = findMaxOnsets(runInfo.get(i));
			
			if(temp > maxOnsets) {
				maxOnsets = temp;
			}
		}
		
		int[] dims = null;
		double[] voxelSize = null;
		double[] origin = null;
		
		// Set dims, vox size, origin from first image file
		try {
			// Get image files path/file name <for read only one image file>
			String dataPath = runInfo.get(0).dataDirectory;
			String fileName = runInfo.get(0).dataFiles.split(" ")[0];
			
			File file = new File (dataPath, fileName);
			Nifti1Dataset image = new Nifti1Dataset(file.toString());
			image.readHeader();
			
			int[] dimsTemp = image.getDimensions();
			dims = new int[]{dimsTemp[0], dimsTemp[1], 1, dimsTemp[2]};
			
			voxelSize = image.getVoxelSize();
			
			float[] qOffset = image.qoffset;
			// take qoffset and calculate origin in voxels
			//  - for each dim, origin = -(qoffset) / (voxsize)
			origin = new double[3];
			for (int i = 0; i < 3; ++i) {
				origin[i] = (double) Math.round(-qOffset[i] / voxelSize[i] + 1); // add one because 1-rel		                                                        
			}	
						
		} catch(Exception ex) {
			throw new Exception("Something bad happened when reading nifti", ex);
		}
		
		// Skipped scans handling
		int[] runIndeces = MLFuncs.removeAll(MLFuncs.range(0, numRuns - 1), MLFuncs.subtract(ignoreRuns, 1));
		
		// Ignored slices handling
		int[] sliceIndeces = MLFuncs.removeAll(MLFuncs.range(0, (int)dims[3] - 1), MLFuncs.subtract(ignoreSlices, 1));
		
		int numVoxels = MLFuncs.product(dims); // unmasked (full 3d vol) num voxels
		int numColumns = numVoxels * windowSize; 
		int[] coords = null;
		
		if(useBrainMask) { // Set coords from mask image
			
			NiftiAnalyzeImage maskImage = null;
			try {
				File file = new File(brainMaskFile);
				maskImage = new NiftiAnalyzeImage(file.getParent(), file.getName());
				
				int[] dimsTemp = maskImage.getDimensions();
				int[] maskDims = new int[]{dimsTemp[0], dimsTemp[1], 1, dimsTemp[2]};
				
				if(!MLFuncs.isEqual(maskDims, dims)) {
					throw new Exception("Dimensions of the data do not match that of the brain mask!");
				}
				double[] brainMask = maskImage.getData(sliceIndeces);
				coords = MLFuncs.findGreaterThan(brainMask, 0);
				
			} catch(Exception ex) {
				throw new Exception("Unable to read mask file " + brainMaskFile, ex);
			}
		} else { 
			coords = MLFuncs.zeros(numVoxels);
		}
		
		Matrix datamat = null;
		Matrix stDatamat = null;
		int[] stEventList = null;
		int[] stEventCount = null;
		
		for(int i : runIndeces) {
			RunInformation thisRunInfo = runInfo.get(i);
			String dataPath = thisRunInfo.dataDirectory;
			String[] fileList = thisRunInfo.dataFiles.split(" ");
			int numFiles = fileList.length;
						
			if(numFiles == 1) { // Data for current run is in 1 4D image file. 
								// Set numFiles by reading 4th array dim size from
				                // this file.
				try {
					URL[] url = NiftiAnalyzeImage.getHeaderURLS(dataPath, new String[]{fileList[0]});
					NiftiFile file = new NiftiFile(url);
					numFiles = file.getHeader().getDim()[4]; // Time frame of a dataset
				} catch(Exception ex) {
					throw new Exception("Unable to read file " + fileList[0] + " from path " + dataPath, ex);
				}
			}
			
			for(int j = 0; j < numConditions; j++) {
				String[] sOnsets = thisRunInfo.onsets.get(j).split(" ");
				
				// Extract the datamat scans that match the current condition
				int rowIndex = 0;
				int numOnsets = sOnsets.length;
				
				Matrix eventDatamat = null;
				if(!singleSubject) {
					// eventDatamat is a 1D row matrix of length (num vox in input 3d vol) * (window size)
					// (i.e. same length as each row in datamat (unmasked)).
					eventDatamat = new Matrix(1, numColumns);
				}
				
				progress.startTask("Creating datamat for run " + (i + 1) + ", condition " + (j + 1), "Datamat creation");
				
				for(int k = 0; k < numOnsets; k++) {
					// Get scan index numbers (0-relative) corresponding to first
					// and last scan in current onset window.
					int startScan = Integer.parseInt(sOnsets[k]) - numSkippedScans;
					int endScan = -1;
					if(isBlockedFmri) {
						int blockSize = Integer.parseInt(thisRunInfo.lengths.get(j).split(" ")[k]);
						endScan = startScan + blockSize - 1;
					} else {
						endScan = startScan + windowSize - 1;
					}
					
					// Get scan index numbers corresponding to current 
					// onset window's normalization reference scan block.
					int baselineStart = startScan;
					int baselineEnd = baselineStart;
					if(!conditionInfo.isEmpty()) {
						int refScanOnset = Integer.parseInt(conditionInfo.get(j)[1]);
						int numRefScans = Integer.parseInt(conditionInfo.get(j)[2]);
						baselineStart += refScanOnset;
						// TODO CHECK and FIX: the following line must be a bug; should be:
						// baselineEnd += refScanOnset + numRefScans - 1; 
						baselineEnd += numRefScans - 1;
					}
					
					if(startScan < 0 || numFiles <= endScan) {
						progress.startTask("Scans " + sOnsets[k] + " for the condition " + (j + 1) + " of run " + (i + 1) + " are not included due to out of bound");
						progress.endTask();
					} else if(baselineStart < 0) {
						progress.startTask("Scans " + sOnsets[k] + " for the condition " + (j + 1) + " of run " + (i + 1) + " are not included due to out of bound baseline");
						progress.endTask();
					} else {
						// Read in image files
						NiftiAnalyzeImage image3D = null;
						Nifti1Dataset image4D = null;
											
						Matrix dataset = new Matrix(endScan - startScan + 1, numVoxels); // still unmasked
						for(int scan = startScan; scan <= endScan; scan++) {
																						
							if(fileList.length > 1) {
								try {
									image3D = new NiftiAnalyzeImage(dataPath, fileList[scan]);
									dataset.setMatrix(scan - startScan, scan - startScan, 0, numVoxels - 1, new Matrix(image3D.getData(sliceIndeces), 1));
								} catch(Exception ex) {
									throw new Exception("Unable to read file " + fileList[scan] + " from path " + dataPath, ex);
								}
							} else {
								try {
									File file = new File (dataPath, fileList[0]);
									image4D = new Nifti1Dataset(file.toString());
									dataset.setMatrix(scan - startScan, scan - startScan, 0, numVoxels - 1, new Matrix(NiftiIO.readNiftiData(image4D.getDataFilename(), scan), 1));
							} catch(Exception ex) {
									throw new Exception("Unable to read file " + fileList[0] + " from path " + dataPath, ex);
								}
							}
						}
						
						// Find brain voxel coords for each onset, and accumulated to find common coords for all onsets
						// (common coords will equal zero at brain voxels; > 0 at non-brain voxels).
						if (!(useBrainMask)) {
							// calculate mask using threshold value
							//
							// coords: add one to each *non-brain* voxel (==> zeros represent mask)
							coords = MLFuncs.sum(coords, findOnsetCoords(dataset, coordThresh, considerAllVoxels));
						}
						else {
							// brain mask file has been supplied 
						}
						
						Matrix tempDatamat = dataset.copy();
						
						if(normalizeMeanVolume) {
							Matrix meanDataset = null; // will contain masked dataset
							if(!useBrainMask) {
								// coords contains zeros at mask voxels when no brain mask file used;
								// mask coord indices are the indices where coords == 0
								meanDataset = MLFuncs.getColumns(dataset, MLFuncs.find(coords, 0));
							} else {
								// coords already contains the mask coord indices when brain mask
								// file used
								meanDataset = MLFuncs.getColumns(dataset, coords);
							}
							meanDataset = MLFuncs.rowMean(meanDataset);
							meanDataset = MLFuncs.replicateColumns(meanDataset.transpose(), numVoxels);
							if(isBlockedFmri) {
								dataset = dataset.arrayRightDivide(meanDataset);
							} else {
								tempDatamat = dataset.arrayRightDivide(meanDataset);
							}
						}
						
						if(isBlockedFmri) {
							tempDatamat = MLFuncs.columnMean(dataset);
						}
						
						// Read in baseline_signals image files
						dataset = new Matrix(baselineEnd - baselineStart + 1, numVoxels);
						for(int scan = baselineStart; scan <= baselineEnd; scan++) {
							if(fileList.length > 1) {
								try {
									image3D = new NiftiAnalyzeImage(dataPath, fileList[scan]);
									dataset.setMatrix(scan - startScan, scan - startScan, 0, numVoxels - 1, new Matrix(image3D.getData(sliceIndeces), 1));
								} catch(Exception ex) {
									throw new Exception("Unable to read file " + fileList[scan] + " from path " + dataPath, ex);
								}
							} else {
								try {
									File file = new File (dataPath, fileList[0]);
									image4D = new Nifti1Dataset(file.toString());
									dataset.setMatrix(scan - baselineStart, scan - baselineStart, 0, numVoxels - 1, new Matrix(NiftiIO.readNiftiData(image4D.getDataFilename(), scan), 1));
								} catch(Exception ex) {
									throw new Exception("Unable to read file " + fileList[0] + " from path " + dataPath, ex);
								}
							}
							
						}
						
						// This is not a duplicate of normalize_volume_mean, it is doing for baseline dataset
						if(normalizeMeanVolume) {
							Matrix meanDataset = null;
							if(!useBrainMask) {
								meanDataset = MLFuncs.getColumns(dataset, MLFuncs.find(coords, 0));
							} else {
								meanDataset = MLFuncs.getColumns(dataset, coords);
							}
							meanDataset = MLFuncs.rowMean(meanDataset);
							meanDataset = MLFuncs.replicateColumns(meanDataset.transpose(), numVoxels);
							dataset = dataset.arrayRightDivide(meanDataset);
						}
						
						Matrix baselineSignals = MLFuncs.columnMean(dataset);
						
						if(!isBlockedFmri) {
							// repmat for win_size of rows
							baselineSignals = MLFuncs.replicateRows(baselineSignals, windowSize);
						}
						// if 0, make it less significant, so it can be removed later
						int[] zeroBaselineIndeces = MLFuncs.find(baselineSignals, 0);
						baselineSignals = MLFuncs.setValues(baselineSignals, zeroBaselineIndeces, 99999);
						if(normalizeSignalMean) {
							tempDatamat = tempDatamat.minus(baselineSignals).times(100).arrayRightDivide(baselineSignals);
						}
						
						tempDatamat = MLFuncs.setValues(tempDatamat, zeroBaselineIndeces, 0);
						
						rowIndex++;
						
						if(singleSubject) {
							if(isBlockedFmri) {
								eventDatamat = MLFuncs.append(eventDatamat, tempDatamat);
							} else {
								eventDatamat = MLFuncs.append(eventDatamat, MLFuncs.reshape(tempDatamat, 1, numColumns));
							}
						} else {
							if(isBlockedFmri) {
								eventDatamat.plusEquals(tempDatamat);
							} else {
								eventDatamat.plusEquals(MLFuncs.reshape(tempDatamat, 1, numColumns));
							}
						}
					}
				}
				
				if(!mergeAcrossRunsFlag && rowIndex == 0) {
					throw new Exception("Error generating spatial-temporal datamat: No onsets for condition " + (j + 1) + " in the run " + (i + 1));
				}
				
				// Generate the spatial/temporal datamat from evt_datamat depending on 'across run' or 'within run'
				if(rowIndex != 0) {
					if(!mergeAcrossRunsFlag) {
						if(singleSubject) {
							stEventList = MLFuncs.append(stEventList, MLFuncs.fillArray(eventDatamat.getRowDimension(), i * numConditions + j));
							datamat = MLFuncs.append(datamat, eventDatamat);
						} else {
							stEventList = MLFuncs.append(stEventList, i * numConditions + j);
							datamat = MLFuncs.append(datamat, eventDatamat.times(1.0 / numOnsets));
						}
					} else {
						int[] indeces = null;
						if(stEventList != null && stEventList.length != 0) {
							indeces = MLFuncs.find(stEventList, j);
						}
						
						if(indeces == null || indeces.length == 0) {
							stEventList = MLFuncs.append(stEventList, j);
							stEventCount = MLFuncs.append(stEventCount, rowIndex);
							
							if(singleSubject) {
								datamat = MLFuncs.append(datamat, new Matrix(MLFuncs.flattenVertically(eventDatamat), 1));
							} else {
								datamat = MLFuncs.append(datamat, eventDatamat);
							}
						} else {
							for(int idx : indeces) {
								stEventCount[idx] += rowIndex;
								
								int n = datamat.getColumnDimension();
								Matrix idxRow = datamat.getMatrix(idx, idx, 0, n - 1).copy(); // Inefficient, not sure how jama handles get (i.e. copy or not)
								if(singleSubject) {
									datamat.setMatrix(idx, idx, 0, n - 1, idxRow.plusEquals(new Matrix(MLFuncs.flattenVertically(eventDatamat), 1)));
								} else {
									datamat.setMatrix(idx, idx, 0, n - 1, idxRow.plusEquals(eventDatamat));
								}
							}
						}
					}
				}
				
				progress.endTask();
			}
		}
			
		progress.startTask("Postprocessing to shape the datamat");
		
		if(!useBrainMask) {
			coords = MLFuncs.find(coords, 0);
		}
		
		if(mergeAcrossRunsFlag && singleSubject) {
			int singleSubjectRows = 0;
			Matrix singleDatamat = null;
			int n = datamat.getColumnDimension();
			for(int j = 0; j < stEventList.length; j++) {
				Matrix temp = datamat.getMatrix(j, j, 0, n - 1).copy().times(1.0 / numRuns);
				singleSubjectRows = n / numColumns;
				temp = MLFuncs.reshape(temp, singleSubjectRows, numColumns);
				singleDatamat = MLFuncs.append(singleDatamat, temp);
			}
			
			datamat = singleDatamat;
			
			stEventList = MLFuncs.flattenVertically(MLFuncs.replicateRows(stEventList, singleSubjectRows));
		}
		
		if(isBlockedFmri) {
			stDatamat = MLFuncs.getColumns(datamat, coords);
		} else {
			int[][] bigCoords = new int[][]{MLFuncs.times(MLFuncs.plus(coords, 1), windowSize)};
			int[] bigCoords2 = bigCoords[0].clone();
			
			for(int i = 0; i < windowSize - 1; i++) {
				bigCoords2 = MLFuncs.subtract(bigCoords2, 1);
				bigCoords = MLFuncs.append(new int[][]{bigCoords2}, bigCoords);
			}
			
			stDatamat = MLFuncs.getColumns(datamat, MLFuncs.flattenVertically(MLFuncs.minus(bigCoords, 1)));
		}
				
		if(mergeAcrossRunsFlag && !singleSubject) {
			int n = stDatamat.getColumnDimension();
			Matrix tmp = new Matrix(stEventList.length,n);
			
			for(int j = 0; j < stEventList.length; j++) {
				Matrix jthRow = stDatamat.getMatrix(j, j, 0, n - 1); // Inefficient, not sure how jama handles get (i.e. copy or not)
				stDatamat.setMatrix(j,j, 0, n - 1, jthRow.times(1.0 / stEventCount[j]));
				tmp.setMatrix(stEventList[j],stEventList[j],0,n-1, stDatamat.getMatrix(j,j,0,n-1)); //Re-order stDatamat to match condition order
			}
			
			stDatamat = tmp;
			stEventList = MLFuncs.sortAscending(stEventList); //Re-order stEventList to match condition order
		}
		
		progress.endTask();
		
		String extension = fMRIDatamatFileFilter.EXTENSION;
		if(isBlockedFmri) {
			extension = BfMRIDatamatFileFilter.EXTENSION;
		}
		String fileName = datamatPrefix + extension;
		
		progress.startTask("Saving to file " + fileName);
				
		try {
			new DatamatSaver(stDatamat, coords, dims, voxelSize, origin, stEventList, windowSize, sessionFile, normalizeMeanVolume, singleSubject, brainMaskFile, coordThresh, considerAllVoxels, numSkippedScans, runIndeces, ignoreSlices, normalizeSignalMean, mergeAcrossRunsFlag, fileName);
		} catch(Exception ex) {
			throw new Exception("Unable to write to file " + fileName, ex);
		}
		
		progress.endTask();
		
		progress.complete();
	}
	 
	/** 
	 * 
	 * @param dataset Matrix of (unmasked) image scan data values; each row corresponds to a
	 *                scan.
	 * @param coordThresh Value between 0 and 1: factor by which to multiply max value
	 *                    in each scan when calculating scan threshold value.
	 * @param considerAllVoxels True if voxels greater than or equal to calculated thresh
	 *                                are to be considered brain voxels;
	 *                          False if only voxels strictly greater than calculated thresh  
	 *                                are to be considered brain voxels.
	 * @return int array (length num voxels (i.e. columns) in input dataset scans) consisting
	 *             of zeros if voxel belongs to AND mask (i.e. intersection) of scan
	 *             thresholded scan masks, and ones otherwise 
	 */
	private int[] findOnsetCoords(Matrix dataset, double coordThresh, boolean considerAllVoxels) {
		int numScans = dataset.getRowDimension();
		int numVoxels = dataset.getColumnDimension();
		
		int[] nonBrainCoords = new int[numVoxels];
		
		for(int i = 0; i < numScans; i++) {
			// Multiply max value in current scan by input threshold  
			// (between 0 and 1) to get current scan thresh
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
