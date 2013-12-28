package pls.analysis;

import java.util.ArrayList;
import java.util.Collections;
import Jama.Matrix;

public class RRIIsLowVariability{

	protected int status = 0;
	
	@SuppressWarnings("unchecked")
	public RRIIsLowVariability(double[] bodat, double[] behavdata) {
		
		//Get what is in this particular behavmat
		double[] behav = behavdata;
		
		ArrayList origU = new ArrayList();
		ArrayList v = new ArrayList();
		
		for(int i = 0; i < behav.length; i++) {
			if(!origU.contains(behav[i])){
				origU.add(behav[i]);
			}
			v.add(bodat[i]);
		}
					
		Collections.sort(origU);
		
		Object[] temp = v.toArray();
		Matrix bodatU = new Matrix(origU.size(),1);
											
		//Get how are those unique values clustered & does any one of them appear too often
		for(int k = 0; k <origU.size(); k++){
				for(int j = 0; j < behav.length; j++){
						if(temp[j].equals(origU.get(k))){
							bodatU.set(k, 0, bodatU.get(k, 0) + 1);
						}		
				}
		}
		
		double max = 0;
					
		//Find the max number of times a values appears
		for(int l = 0; l < bodatU.getRowDimension(); l ++) {
			if(bodatU.get(l,0) > max){
				max = bodatU.get(l,0);
			}
		}
				
		if (max * 1.0/ behav.length >= 0.5){
			status = 1;
		}
	}
	
}
