package pls.analysis;

import java.util.Map;
import java.util.Vector;

import pls.shared.MLFuncs;
import Jama.Matrix;

import com.jmatio.io.MatFileFilter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;

import extern.NewMatFileReader;

public class ConcatenateFmriBehavioralDatamat extends ConcatenateDatamat {
	
	public ConcatenateFmriBehavioralDatamat(Vector<String[]> sessionProfiles, int[] conditionSelection, String behaviorFilename) throws Exception {
		super(sessionProfiles, conditionSelection);
		
		loadBehavDataList(behaviorFilename);

	}
	
	public void stackDatamat() throws Exception {
		totalEvents = MLFuncs.sum(subjectGroup) * numConditions;
		datamat = new Matrix(totalEvents, winSize * numVoxels);
		int firstRow = 0;
		int[] firstConditionOrder = null;
		
		int count = 0;
		// Go through each subject which is represented by each profile
		for(int i = 0; i < numGroups; i++) {
			
			Matrix groupTempNewStDatamat = new Matrix(0, winSize * numVoxels);
			int[] groupTempNewEventList = null;
			int[] groupFirstConditionOrder = null;
			
			for(int j = 0; j < sessionGroup[i]; j++, count++) {
				
				String datamatFilename = getDatamatFilename(count);
				Map<String, MLArray> datamatMap = new NewMatFileReader(datamatFilename, new MatFileFilter(new String[]{"st_datamat"})).getContent();
				
				double[][] stDatamat = ((MLDouble)datamatMap.get("st_datamat")).getArray();
				
				int[] thisCoords = getStCoords(count);
				thisCoords = MLFuncs.subtract(thisCoords, 1);  // make coords 0-relative
				int[] coordIdx = MLFuncs.find(MLFuncs.getItemsAtIndices(M, thisCoords), numProfiles);
				
				int numRuns = getEventList(count).length;

				double createVer = getCreateVer(count);
				
				int lastRow = numRuns + firstRow - 1;
				
				int[] behavMask = getBehavMask(stDatamat);
				
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
				double[][] tempDatamat = stDatamat;
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
				
				// Intentionally reverse the order to each condition in each run, if behavpls
				// because in behavpls, the whole thing will be then re-order again
				tempNewStDatamat = MLFuncs.getRows(tempNewStDatamat, behavMask);
				tempEventList = MLFuncs.getItemsAtIndices(tempEventList, behavMask);
				
				groupTempNewStDatamat = MLFuncs.append(groupTempNewStDatamat, new Matrix(tempNewStDatamat));
				groupTempNewEventList = MLFuncs.append(groupTempNewEventList, tempEventList);
				datamat.setMatrix(firstRow, lastRow, 0, winSize * numVoxels - 1, new Matrix(tempNewStDatamat));
				
				eventList = MLFuncs.append(eventList, getEventList(count));
				
				firstRow = lastRow + 1;
			}
			
			int[] groupSubjectMask = null;
			
			for(int ii = 0; ii < numConditions; ii++) {
				groupSubjectMask = MLFuncs.append(groupSubjectMask, MLFuncs.find(groupFirstConditionOrder, 1));
				int end = groupFirstConditionOrder.length - 1;
				int[] lastItem = new int[]{groupFirstConditionOrder[end]};
				groupFirstConditionOrder = MLFuncs.append(lastItem, MLFuncs.getItemsAtIndices(groupFirstConditionOrder, MLFuncs.range(0, end - 1)));
			}
			
			groupTempNewStDatamat = MLFuncs.getRows(groupTempNewStDatamat, groupSubjectMask);
			groupTempNewEventList = MLFuncs.getItemsAtIndices(groupTempNewEventList, groupSubjectMask);
			
			newDataList.add(groupTempNewStDatamat);
			numSubjectList = MLFuncs.append(numSubjectList, MLFuncs.sum(groupFirstConditionOrder));
			eventListList.add(groupTempNewEventList);
			
		}
		
		numBehavSubj = MLFuncs.sum(firstConditionOrder);
	}
	
	public void deselectConditions() {
		for(int i = 0; i < eventListList.size(); i++) {
			int[] tempNewEventList = eventListList.get(i);
			MaskEventList fmel = new MaskEventList(tempNewEventList, conditionSelection);
			
			tempNewEventList = fmel.eventList;
			mask = fmel.mask;
			eventLength = fmel.eventLength;
			
			eventListList.setElementAt(tempNewEventList, i);
			datamat = newDataList.get(i);
			newDataList.setElementAt(MLFuncs.getRows(datamat, mask), i);
		}
		
		datamat = null;
		eventList = null;
		
		for(int i = 0; i < numGroups; i++) {
			datamat = MLFuncs.append(datamat, newDataList.get(i));
			eventList = MLFuncs.append(eventList, eventListList.get(i));
		}
	}
	
	/**
	 * Stack behavdata and get behavmask (re-order for each session file
	 * to make it 'each condition in each run (yes, reversed)'
	 */
	private int[] getBehavMask(double[][] stDatamat) {
		int m = stDatamat.length;
		int[] behavMask1 = MLFuncs.range(0, m - 1);
		int nrr = m / numConditions;
		int[][] behavMask2 = MLFuncs.reshape(new int[][]{behavMask1}, nrr, numConditions);
		int[][] behavMask3 = MLFuncs.transpose(behavMask2);
		return MLFuncs.flattenHorizontally(behavMask3);
	}
	
	private void loadBehavDataList(String behaviorFilename) throws Exception {
		double[][] data = MLFuncs.load(behaviorFilename);
		if(data == null) {
			throw new Exception("Behaviour data is corrupt");
		}
		
		behavData = new Matrix(data);
		
		int currentRow = 0;
		for(int i = 0; i < numGroups; i++) {
			int currBehavDataSize = subjectGroup[i] * numConditions - 1;
			int[] currBehavDataRange = MLFuncs.range(currentRow, currentRow + currBehavDataSize);
			behavDataList.add(MLFuncs.getRows(behavData, currBehavDataRange));
			currentRow += currBehavDataSize + 1;
		}
	}
}
