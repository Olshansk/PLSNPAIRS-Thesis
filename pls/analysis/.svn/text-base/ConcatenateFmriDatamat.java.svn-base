package pls.analysis;

import java.util.Map;
import java.util.Vector;

import javax.swing.JProgressBar;

import pls.shared.MLFuncs;
import Jama.Matrix;

import com.jmatio.io.MatFileFilter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;

import extern.NewMatFileReader;

public class ConcatenateFmriDatamat extends ConcatenateDatamat {
	
	public ConcatenateFmriDatamat(Vector<String[]> sessionProfiles, String currentDirectory, int[] conditionSelection, JProgressBar progressBar) throws Exception {
		super(sessionProfiles, currentDirectory, conditionSelection, progressBar, null, null);
	}
	
	public ConcatenateFmriDatamat(Vector<String[]> sessionProfiles, int[] conditionSelection) throws Exception {
		super(sessionProfiles, conditionSelection);
	}
	
	public void stackDatamat() throws Exception {
		datamat = null;
		int firstRow = 0;
		int[] firstConditionOrder = null;
		
		int count = 0;
		double[][] datamatArray = null;
		// Go through each subject which is represented by each profile
		for(int i = 0; i < numGroups; i++) {
			
			double[][] groupTempNewStDatamat = null;
			int[] groupTempNewEventList = null;
			int[] groupFirstConditionOrder = null;
			
			for(int j = 0; j < sessionGroup[i]; j++, count++) {
				
				int[] thisCoords = getStCoords(count);
				thisCoords = MLFuncs.subtract(thisCoords, 1);  // make coords 0-relative
				int[] coordIdx = MLFuncs.find(MLFuncs.getItemsAtIndices(M, thisCoords), numProfiles);
				
				int numRuns = getEventList(count).length;

				double createVer = getCreateVer(count);
				
				int lastRow = numRuns + firstRow - 1;
				
				// get subj_name for designPLS
				int[] thisSubjOrder = new int[numRuns];
				
				int firstCond = 0;
				int jj = 0;
				
				while(firstCond < numRuns) {
					thisSubjOrder[firstCond] = 1;
					firstCond += numConditions;
					if(createVer < 4.0512201) {
						subjectName.add("Subj" + (count + 1) + "Run" + (jj + 1));
					} else {
						subjectName.add("Subj" + (count + 1));
					}
					jj++;
				}
				
				firstConditionOrder = MLFuncs.append(firstConditionOrder, thisSubjOrder);
				groupFirstConditionOrder = MLFuncs.append(groupFirstConditionOrder, thisSubjOrder); // For behav pls
				
				// Stack datamat
				String datamatFilename = getDatamatFilename(count);
				Map<String, MLArray> datamatMap = new NewMatFileReader(datamatFilename, new MatFileFilter(new String[]{"st_datamat"})).getContent();
				
				double[][] tempDatamat = ((MLDouble)datamatMap.get("st_datamat")).getArray();
				//double[][][] tempDatamat2 = MLFuncs.reshape(tempDatamat, numRuns, winSize, thisCoords.length);
				//double[][] tempNewStDatamat = MLFuncs.reshape(MLFuncs.getXYArrays(tempDatamat2, coordIdx), numRuns, winSize * numVoxels);
				
				double[][] tempNewStDatamat = new double[numRuns][winSize * numVoxels];
				
				for(int i2 = 0; i2 < numRuns; i2++) {
					int count2 = 0;
					for(int j2 = 0; j2 < numVoxels; j2++) {
						for(int k2 = 0; k2 < winSize; k2++, count2++) {
							tempNewStDatamat[i2][count2] = tempDatamat[i2][(winSize * coordIdx[j2]) + k2];
						}
					}
				}
				
				int[] tempEventList = getEventList(j);
				
				groupTempNewStDatamat = MLFuncs.append(groupTempNewStDatamat, tempNewStDatamat);
				groupTempNewEventList = MLFuncs.append(groupTempNewEventList, tempEventList);
				
				firstRow = lastRow + 1;
			}
			//int[] sortedIndex = MLFuncs.getSortedIndex(groupTempNewEventList);
			//int[] sortedIndex = {0, 7, 14, 1, 8, 15, 2, 9, 16, 3, 10, 17, 4, 12, 18, 5, 13, 20, 6, 11, 19};
			int[] sortedIndex = new int[groupTempNewEventList.length];
			int index = 0;
			for (int m = 0; m != numConditions; m++) {
				for (int n = 0; n != sessionGroup[i]; n++) {
					sortedIndex[index] = m + (numConditions * n);
					index++;
				}
			}
			int[] groupTempNewEventList3 = MLFuncs.getItemsAtIndices(groupTempNewEventList, sortedIndex);
			
			eventList = MLFuncs.append(eventList, groupTempNewEventList3);
			datamatArray = MLFuncs.append(datamatArray, MLFuncs.getRows(groupTempNewStDatamat, sortedIndex));
		}
		
		datamat = new Matrix(datamatArray);
		numBehavSubj = MLFuncs.sum(firstConditionOrder);
	}
}
