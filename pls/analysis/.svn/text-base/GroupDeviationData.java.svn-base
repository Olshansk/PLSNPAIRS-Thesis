package pls.analysis;

import java.util.Vector;

import pls.shared.MLFuncs;
import Jama.Matrix;

public class GroupDeviationData {
	
	protected Matrix data = null;
	protected Matrix sMeanmat = null;
	
	protected Matrix design = null;
	
	int numGroups = 0;

	public GroupDeviationData(Matrix stDatamat, int numConditions, int[] eventList, int[] subjectGroup) {
		numGroups = subjectGroup.length;
		
		int gEndIdx = 0;
		
		int n = stDatamat.getColumnDimension();
		
		for(int g = 1; g <= numGroups; g++) {
			int gStartIdx = gEndIdx + 1;
			gEndIdx = gStartIdx + subjectGroup[g - 1] * numConditions - 1;
			int[] gRange = MLFuncs.range(gStartIdx - 1, gEndIdx - 1);
			
			int[] gEventList = new int[eventList.length];
			gEventList = MLFuncs.setValues(gEventList, gRange, eventList);
			
			Vector<int[]> taskIdx = new Vector<int[]>(numConditions);
			
			for(int i = 0; i < numConditions; i++) {
				taskIdx.add(MLFuncs.find(gEventList, i + 1));
			}
			
			Matrix meanDatamat = new Matrix(numConditions, n);
			
			for(int i = 0; i < numConditions; i++) {
				meanDatamat.setMatrix(i, i, 0, n - 1, MLFuncs.columnMean(MLFuncs.getRows(stDatamat, taskIdx.get(i))));
			}
			
			Matrix ones = new Matrix(numConditions, 1, 1);
			Matrix mean = MLFuncs.columnMean(meanDatamat);
			Matrix groupDatamat = meanDatamat.minus(ones.times(mean));
			data = MLFuncs.append(data, groupDatamat);
			
			//Finding grp_smeanmat (labeled smeanmat)
			Matrix tmp = stDatamat.getMatrix(gRange, 0, stDatamat.getColumnDimension()-1);
			Matrix tmp_ones = new Matrix (tmp.getRowDimension(),1,1);
			Matrix smeanmat = tmp.minus(tmp_ones.times(mean));
			sMeanmat = MLFuncs.append(sMeanmat, smeanmat);
							
			}
	}

	public GroupDeviationData(Matrix stDatamat, int numConditions, int[] eventList, int[] subjectGroup, Matrix design) {

		numGroups = subjectGroup.length;
		
		int gEndIdx = 0;
		
		int n = stDatamat.getColumnDimension();
		
		for(int g = 1; g <= numGroups; g++) {
			int gStartIdx = gEndIdx + 1;
			gEndIdx = gStartIdx + subjectGroup[g - 1] * numConditions - 1;
			int[] gRange = MLFuncs.range(gStartIdx - 1, gEndIdx - 1);
			
			int[] gEventList = new int[eventList.length];
			gEventList = MLFuncs.setValues(gEventList, gRange, eventList);
			
			Vector<int[]> taskIdx = new Vector<int[]>(numConditions);
			
			for(int i = 0; i < numConditions; i++) {
				taskIdx.add(MLFuncs.find(gEventList, i + 1));
			}
			
			Matrix meanDatamat = new Matrix(numConditions, n);
			
			for(int i = 0; i < numConditions; i++) {
				meanDatamat.setMatrix(i, i, 0, n - 1, MLFuncs.columnMean(MLFuncs.getRows(stDatamat, taskIdx.get(i))));
			}
			
			//Need to change the codes here also, missing the grp_smeanmat;
			
			Matrix groupDatamat = meanDatamat;
			data = MLFuncs.append(data, groupDatamat);
			
			int[] span = MLFuncs.range((g - 1) * numConditions, (g * numConditions) - 1);
			this.design = MLFuncs.append(this.design, MLFuncs.getRows(design, span));
		}
	}
	//added for PET
	public GroupDeviationData(ConcatenateDatamat st,  Matrix pdesign){
		numGroups = st.newDataList.size();
		int numCond = st.num_cond_lst[0];
		int k, n;
		Matrix datamat;
		//Matrix stackedDatamat = null;
		Matrix stackedDesigndata = null;
		Matrix stackedData = null;
		Matrix p_data = null;;
		
		for(int i=1; i<=numGroups; i++)
		{
			k = st.num_cond_lst[i-1];
			n = st.numSubjectList[i-1];
			
			datamat = st.newDataList.elementAt(i-1);
			p_data = new RRITaskMean(datamat,n).taskMean;
			
			//if more than one group, stacked data together
			//st.datamat is already contains stacked_datamat info it is founded at ConcatenatePETDatamat class stackData function  
			//there is no need to clalculate it here again
			//stackedDatamat = MLFuncs.append(stackedDatamat, datamat);
			
			stackedData = MLFuncs.append(stackedData, p_data);
			/* in function findDesign design is founded
			int begin = (i)*numCond; // in matlab (i-1)*num_cond+1 index begin form 1 
			int end = ((i+1)*numCond)-1; // in malab i*num_cond index begin from 1
			Matrix tmp = MLFuncs.getRows(design, MLFuncs.range(begin, end));
			//design used instead of stacked_designdata 
			stackedDesigndata = MLFuncs.append(stackedDesigndata, tmp);
			*/
		}
		data = stackedData; 
		//design = stackedDesigndata;
		findDesign(numCond, pdesign);
		
		
	}

	//added for PET
	public GroupDeviationData(ConcatenateDatamat st, Matrix data_p, Matrix pdesign){
		Matrix stackeDataP = null;
		numGroups = st.newDataList.size();
		int numCond = st.num_cond_lst[0];
		int k, n;
		for(int i=1; i<=numGroups; i++)
		{
			k = st.num_cond_lst[i-1];
			n = st.numSubjectList[i-1];

			//???????
			//int span = MLFuncs.sum(MLFuncs.getItemsAtIndeces(st.numSubjectList, MLFuncs.range(0, i-2)))*numCond;//??????????????
			int span = MLFuncs.sum(MLFuncs.getItemsAtIndices(st.numSubjectList, MLFuncs.range(0, i-1)))*numCond;//group length

			if(numGroups == 1)
				data = new RRITaskMean(data_p, n).taskMean;
			else{
				int begin = span; // 1+span
				int end = n*k+span-1; //n*k+span
				Matrix tmp = MLFuncs.getRows(data_p, MLFuncs.range(begin, end));
				data = new RRITaskMean(tmp, n).taskMean;
			}
			stackeDataP = MLFuncs.append(stackeDataP, data);
		}
		data = stackeDataP;
		findDesign(numCond, pdesign);
	}
	void findDesign(int numCond, Matrix pdesign){
		Matrix stackedDesigndata=null;
		
		for(int i=1; i<=numGroups; i++)
		{
			int begin = (i-1)*numCond;  //in matlab (i-1)*numCond+1
			int end = i*numCond-1; // in malab i*num_cond 
			Matrix tmp = MLFuncs.getRows(pdesign, MLFuncs.range(begin, end));
			//design used instead of stacked_designdata 
			stackedDesigndata = MLFuncs.append(stackedDesigndata, tmp);
		}
		design = stackedDesigndata;
	}
}
