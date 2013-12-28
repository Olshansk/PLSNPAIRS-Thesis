package pls.analysis;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Map;
import java.util.StringTokenizer;
import java.util.Vector;

import pls.shared.MLFuncs;
import Jama.Matrix;

import com.jmatio.io.MatFileFilter;
import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;

public class ConcatenatePetDatamat extends ConcatenateDatamat {
	
	//datamat profiles incluses datamat group file list
	public ConcatenatePetDatamat(Vector<String[]> datamatProfiles, int[] conditionSelection, String behaviorFilename) throws Exception {
		super(1, datamatProfiles, conditionSelection);
		
		loadBehavDataList(behaviorFilename);
	}
	public void stackDatamat() throws Exception {
		dims = getPetDims(0);
		int size = dims[0] * dims[1] * dims[2] * dims[3];
		
		System.out.println("size:"+size+"");

		double  [] multed = new double[size];
		
		for(int k=0; k<coords.length; k++){
			multed[coords[k]]= 1;
		}
		
		//funcw(multed, "jresmulted.txt");
		
		
		//double [] relative = new double[multed.length];
		double [] relative=null;
		int [] relativeindex; 
		
		num_cond_lst = new int[numProfiles];
		numSubjectList = new int[numProfiles];
		System.out.println("\ncoord_idx r, c"+coord_idx.getRowDimension()+","+coord_idx.getColumnDimension());
		
		for(int i=0; i<numProfiles; i++){
			relative = new double[coord_idx.getColumnDimension()];
			//relative location needs both coord_idx and multed information
			for(int j=0; j<multed.length;j++){
				//System.out.println("\nyapti"+i+","+j);	
				relative[j]= coord_idx.get(i,j)+ multed[j];
			}
			//remove unrelated information
			relativeindex = MLFuncs.findGreaterThan(relative, 0.0);
			
			relative = MLFuncs.getItemsAtIndices(relative, relativeindex);
			
			//funcw(relative, "jresrelative"+i+".txt");

			//find relative location of new coords
			relativeindex = MLFuncs.find(relative, 2);
			
			//funcw(relativeindex, "jresrelativefind"+i+".txt");

			int numSubjs = getNumSubject(i);
			int numCond = getNumCondition(i);
			
			//num_cond_lst.addElement(new Integer(numCond));
			num_cond_lst[i] = MLFuncs.sum(conditionSelection);
			numSubjectList[i] = numSubjs;
			
			System.out.println("datamat requested at "+i);
			int [] petcoords = getPetCoords(i);
			
			datamat = getDatamat(i);
			//datamat = datamatList.elementAt(i);
			
			//funcw(datamat, "jresdatamatstack"+i+".txt");
			
			int [] selectedSubjects = MLFuncs.ones(numSubjs);
			
			Matrix bmask = new Matrix(selectedSubjects.length,conditionSelection.length);

			for(int k=0; k<selectedSubjects.length;k++){
				for(int j=0; j<conditionSelection.length;j++){
					bmask.set(k, j,selectedSubjects[k]*conditionSelection[j]);
				}
			}
			
			double []tempbmask = MLFuncs.flattenVertically(bmask);
			int [] bmaskindex = MLFuncs.findGreaterThan(tempbmask, 0.0);
			
			//funcw(bmaskindex, "jresbmaskdatamat.txt");
			
			double[][] tempNewDatamat = new double[bmaskindex.length][relativeindex.length];
			
			for(int k=0; k<bmaskindex.length; k++){
				for(int j=0; j<relativeindex.length;j++){
					tempNewDatamat[k][j] = datamat.get(bmaskindex[k], relativeindex[j]);
				}
			}

			//newdataLst array of matrix gibi birsey olmali append degil
			newDataList.addElement(new Matrix(tempNewDatamat));

			//funcw(newDataList.elementAt(i), "jresnewdatalist.txt");
			//filecompare("resnewdatalst.txt","jresnewdatalist.txt","c:\\nese\\datas\\newdatalist.txt");
			
			if(newDataList.elementAt(i).getRowDimension()==0) {
				throw new Exception("Merged datamat is empty!");
			}
			
			//stackedDatamat
			
			//datamat = MLFuncs.append(datamat, newDataList.get(i));

			///??? this part never done avg = 0, and never becomes this part as turue
			//why it is written this time to the file pet_get_common.m 
			/*
			 * 
			double avg =0;
			int[][] dat = null;
			Matrix gm = null;

			 if(avg!=0){
				//average on condition for each newDataLst
				//dat = MLFuncs.reshape(new int[][]{newdataLst}, numSubjs, numCond);
				System.out.println("1. row uzunluk"+dat[0].length+ "row uzunluk "+dat.length);
				Matrix d = new Matrix(dat.length, dat[0].length);
				for(int k=0; k<dat.length; k++){
					for(int j=0; j<dat[0].length;j++){
						d.set(k, j, dat[k][j]);		
					}
				}
				gm = MLFuncs.rowMean(d);
			}
			//??? this part never done avg = 0, and never becomes this part as turue
			// why it is written this time to the file pet_get_common.m 
			
			if(avg!=0)
			{
				//fill all the conditions with grand mean
				
			}
			*/
		}
		checkValid();
		
	}
	private void checkValid() throws Exception {
	
		System.out.println("Pet check valid");
		for(int i=0; i<numProfiles; i++){
			
			int m = newDataList.elementAt(i).getRowDimension();
			int n = newDataList.elementAt(i).getColumnDimension();
			
			if(newDataList.elementAt(i) == null || m == 0 || n == 0 || m > n) {
				System.out.println(i+"newdatalist r"+m+", c"+n);
				throw new Exception("Invalid datamat");
			}
		}
	}
		
	
	public void deselectConditions() {
		/*
		 * 
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
		*/
		
		//stackedDatamat
		datamat = null;
		//eventList = null;
		
		for(int i = 0; i < numGroups; i++) {
			datamat = MLFuncs.append(datamat, newDataList.get(i));
			//eventList = MLFuncs.append(eventList, eventListList.get(i));
		}
		//numSubjectList = sessionGroup.clone();
		 
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
			return;
			//throw new Exception("Behaviour data is corrupt");
		}
		System.out.println("begin of loadBehaveDatalist function");
		
		behavData = new Matrix(data);
		int currentRow = 0;
		for(int i = 0; i < numGroups; i++) {
			int currBehavDataSize = subjectGroup[i] * numConditions - 1;
			int[] currBehavDataRange = MLFuncs.range(currentRow, currentRow + currBehavDataSize);
			behavDataList.add(MLFuncs.getRows(behavData, currBehavDataRange));
			currentRow += currBehavDataSize + 1;
		}
		System.out.println("end of loadBehaveDatalist function");
		
	}
}
