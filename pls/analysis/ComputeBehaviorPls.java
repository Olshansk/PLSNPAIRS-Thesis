package pls.analysis;

import java.io.*;
import java.util.*;

import java.util.Vector;

import Jama.Matrix;
import Jama.SingularValueDecomposition;
import pls.shared.MLFuncs;

public class ComputeBehaviorPls {
	
	/*added for PET*/
	protected Matrix singleCondLst = null;
	protected Matrix stackedDatamat = null; 
	protected int [] numLowVariabilityBehavBoots = null;
	/*******************/
	
	protected Vector<Matrix> behavDataList = new Vector<Matrix>();
	
	protected Matrix brainLV = null;
	
	protected Matrix S = null;
	
	protected Matrix behavLV = null;
	
	protected Matrix brainScores = null;
	
	protected Matrix behavScores = null;
	
	protected Matrix lvCorrs = null;
	
	protected Vector<Matrix> datamatCorrsList = new Vector<Matrix>();
	
	protected Matrix stackedBehavData = null;
	
	protected Matrix origPost = null;
	
	//constructor for fMRI
	public ComputeBehaviorPls(Matrix stackedDatamat, Matrix behavData, Vector<Matrix> behavDataList, Vector<Matrix> newDataList, int[] numSubjectList, int numConditions) {
		
		Matrix stackedDatamatCorrs = null;
		stackedBehavData = behavData.copy();
		this.behavDataList = behavDataList;
		
		int numGroups = newDataList.size();
		
		int k = numConditions;
		
		// loop accross the groups, and calculate datamatcorrs for each group
		for(int i = 0; i < numGroups; i++) {
			int n = numSubjectList[i];
			Matrix datamat = newDataList.get(i);
			// Compute correlation
			Matrix datamatCorrs = new RRICorrMaps(behavDataList.get(i), datamat, n, k).maps;
			stackedDatamatCorrs = MLFuncs.append(stackedDatamatCorrs, datamatCorrs);
			datamatCorrsList.add(datamatCorrs);
		}
		
		SingularValueDecomposition USV = new SingularValueDecomposition(stackedDatamatCorrs.transpose());
		
		brainLV = USV.getU();
		S = MLFuncs.diag(USV.getS()).transpose();
		behavLV = USV.getV();

		// Calculate behav scores
		RRIGetBehaviorScores rgb = new RRIGetBehaviorScores(stackedDatamat, stackedBehavData, brainLV, behavLV, numConditions, numSubjectList);
		brainScores = rgb.scores;
		behavScores = rgb.fScores;
		lvCorrs = rgb.lvCorrs;
	}

	//constructor for PET and fMRI
	public ComputeBehaviorPls(int imagingType, ConcatenateDatamat st, boolean isbehav) {
		
		Matrix stackedDatamatCorrs = null;
		int k=0;
		if(imagingType != 2){ // if it is not PET
			stackedBehavData = st.behavData.copy();
			k = st.numConditions;
		}
		this.behavDataList = st.behavDataList;
		
		int numGroups = st.newDataList.size();
		
		if(imagingType == 2) // if it is PET
		{
			Matrix tmp=null;
			if(isbehav==false && st.num_cond_lst[0]==0)
			{
				for(int g=0; g<numGroups; g++){
					tmp= MLFuncs.append(tmp, st.newDataList.get(g));
				}
			}
			singleCondLst= tmp; // check this part is it true or it has the same meaing with matlab code
		}
		
		// loop accross the groups, and calculate datamatcorrs for each group
		for(int i = 0; i < numGroups; i++) {
			int n = st.numSubjectList[i];
			Matrix datamat = null;
			Matrix datamatCorrs = null;
			
			if(imagingType == 2) // if it is PET
			{
				k = st.num_cond_lst[i];
				if(singleCondLst == null) {
					System.out.println("single cond_list empty");
					datamat = st.newDataList.get(i);
				}
				else if (i==0){
					System.out.println("i=0");
					datamat = singleCondLst;  // check this part is it true or ithas the same meaing with matlab code
				}
			}else{//it is fmri
				datamat = st.newDataList.get(i);
				datamatCorrs = new RRICorrMaps(behavDataList.get(i), datamat, n, k).maps;
			}
					
			if(imagingType == 2) // if it is PET
			{
				//is task PLS
				if(isbehav == false) // it is Task PLS
				{
					if(singleCondLst == null){					
						datamatCorrs = new RRITaskMean(datamat, n).taskMean.minus(new Matrix(k, 1, 1).times(MLFuncs.columnMean(datamat)));
						System.out.println("\nburda 1 n "+n);
					}else if(i==0){
						datamatCorrs = new RRITaskMean(datamat, st.numSubjectList).taskMean.minus(new Matrix(numGroups, 1, 1).times(MLFuncs.columnMean(datamat)));
						System.out.println("\nburda 2");
					}
				}else{//is Behavior PLS
					datamatCorrs = new RRICorrMaps(behavDataList.get(i), datamat, n, k).maps;
					System.out.println("\nburda 3");
				}
			
				//if more tnan one group stack data together
				if(isbehav == true) // is it means if(~isbehav) 
				{
					stackedBehavData = MLFuncs.append(stackedBehavData, behavDataList.elementAt(i));
				}
				if(singleCondLst == null || i==0)
				{
					stackedDatamat = MLFuncs.append(stackedDatamat,datamat);
					stackedDatamatCorrs = MLFuncs.append(stackedDatamatCorrs, datamatCorrs);
					datamatCorrsList.add(datamatCorrs);
				}
			}
			else
			{
				stackedDatamatCorrs = MLFuncs.append(stackedDatamatCorrs, datamatCorrs);
				datamatCorrsList.add(datamatCorrs);
			}
		}
		
		SingularValueDecomposition USV = new SingularValueDecomposition(stackedDatamatCorrs.transpose());
		
		//st.filecompare2("c:\\nese\\datas\\jresstackdatamatcorrs.txt", "resstackdatamatcorrs.txt", "c:\\nese\\datas\\comparestackdatamatcorrs.txt");
		
		brainLV = USV.getU();
		S = MLFuncs.diag(USV.getS()).transpose();
		behavLV = USV.getV();
		
				
		//st.filecompare2("jresbrainlv.txt", "resbrainlv.txt", "comparebrainlv.txt");
/*
		System.out.println("\nDisplay brainLV r" + brainLV.getRowDimension()+", c "+brainLV.getColumnDimension());
		for(int m=0; m<brainLV.getRowDimension(); m++){
			System.out.println("");
			for(int a=0; a<brainLV.getColumnDimension();a++)
				System.out.print(brainLV.get(m,a)+" ");
		}
		
		System.out.println("\nDisplay behavLV r" + behavLV.getRowDimension()+", c "+behavLV.getColumnDimension());
		for(int m=0; m<behavLV.getRowDimension(); m++){
			System.out.println("");
			for(int a=0; a<behavLV.getColumnDimension();a++)
				System.out.print(behavLV.get(m,a)+" ");
		}
		*/
		// Calculate behav scores
		RRIGetBehaviorScores rgb = null;
		
		if(imagingType == 2) // if it is PET
		{
			if(isbehav == false) // is Task PLS
			{
				brainScores = stackedDatamat.times(brainLV);
	
				//Here, behavlv is actually designlv
			    //according to taskpls.m: fscores=design=testvec(designlv)
				// so, designscores = designlv
				int numCol = behavLV.getColumnDimension();
				//expand the num_subj for each row (cond)
				//did the samething testvec
				Matrix tmp=null;
				for(int i=1; i<=st.numGroups;i++){
					k = st.num_cond_lst[i-1];
					int n = st.numSubjectList[i-1];
					int start = (i-1)*k;
					int end = (i-1)*k+k-1;
				
					int[] range = MLFuncs.range(start, end);
		
					tmp = MLFuncs.reshape(MLFuncs.getRows(behavLV, range), 1, numCol * k);

					System.out.println("\nDisplay tmp after resahpe r" + tmp.getRowDimension()+", c "+tmp.getColumnDimension());
					for(int m=0; m<tmp.getRowDimension(); m++){
						System.out.println("\n");
						for(int a=0; a<tmp.getColumnDimension();a++)
							System.out.print(tmp.get(m,a)+" ");
					}
					
					tmp = MLFuncs.replicateColumns(tmp,n); //expand to num_subj
					
					System.out.println("\nDisplay tmp after repmat r" + tmp.getRowDimension()+", c "+tmp.getColumnDimension());
					for(int m=0; m<tmp.getRowDimension(); m++){
						System.out.println("\n");
						for(int a=0; a<tmp.getColumnDimension();a++)
							System.out.print(tmp.get(m,a)+" ");
					}

					tmp = MLFuncs.reshape(tmp, n*k, numCol);
					
					System.out.println("\nDisplay tmp after resahpe r" + tmp.getRowDimension()+", c "+tmp.getColumnDimension());
					for(int m=0; m<tmp.getRowDimension(); m++){
						System.out.println("\n");
						for(int a=0; a<tmp.getColumnDimension();a++)
							System.out.print("tmp value:"+tmp.get(m,a)+" ");
					}
					behavScores = MLFuncs.append(behavScores, tmp);//stack by groups
				}
				
				System.out.println("\nDisplay behavscores r" + behavScores.getRowDimension()+", c "+behavScores.getColumnDimension());
				for(int m=0; m<behavScores.getRowDimension(); m++){
					System.out.println("\n");
					for(int n=0; n<behavScores.getColumnDimension();n++)
						System.out.print(behavScores.get(m,n)+" ");
				}
			}
			else //is behavior PLS
			{	rgb = new RRIGetBehaviorScores(stackedDatamat, stackedBehavData, brainLV, behavLV, st.num_cond_lst[0], st.numSubjectList);
			
				brainScores = rgb.scores;
				behavScores = rgb.fScores;
				lvCorrs = rgb.lvCorrs;
			}
		}
		else{
			rgb = new RRIGetBehaviorScores(st.datamat, stackedBehavData, brainLV, behavLV, st.numConditions, st.numSubjectList);
			brainScores = rgb.scores;
			behavScores = rgb.fScores;
			lvCorrs = rgb.lvCorrs;
		}
		
	}
	public void filecompare(String fj, String fm, String res){
		try {
			BufferedReader readerj = new BufferedReader(new FileReader(fj));
			BufferedReader readerm = new BufferedReader(new FileReader(fm));
			BufferedWriter writer = new BufferedWriter(new FileWriter(res));
			String linej = null;
			String linem = null;
		
			boolean flag = false;
			int countj=0;
			int countm=0;
			int countlj=0;
			int countlm=0;
			
	        while (((linej=readerj.readLine()) != null) && ((linem=readerm.readLine()) != null)){
	        	StringTokenizer stj = new StringTokenizer(linej,",");
	        	StringTokenizer stm = new StringTokenizer(linej,",");
	        	countlj++;
	        	countlm++;
	        	countj=0;
				countm=0;
	        	while(stj.hasMoreTokens())
	        	{
	        		countj++;
		        	countm++;
	        		if(! stj.nextElement().equals(stm.nextElement())){
	        			writer.write("ayni Line"+countlj+", "+countlm+"colum "+countj+", "+countm+"\n");
	        		}
	        		else{
	        		writer.write("farkli Line"+countlj+", "+countlm+"colum "+countj+", "+countm+"\n");
	        		}
	        	}
	        }

        } catch (IOException e) {
            System.err.println(e);
            System.exit(1);
        }
	}
	
}
