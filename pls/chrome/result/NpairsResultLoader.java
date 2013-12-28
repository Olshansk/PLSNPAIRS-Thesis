package pls.chrome.result;

import java.util.ArrayList;

import pls.chrome.result.model.BrainData;
import pls.chrome.result.model.NPairsResultModel;
import pls.shared.MLFuncs;

import com.jmatio.types.MLArray;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLStructure;

public class NpairsResultLoader extends ResultLoader {
	
	public NPairsResultModel mNPairsResultModel;

	public NpairsResultLoader(String filename) {
		super(filename);
	}

	protected void addOtherRelevantFields() {
		mRelevantFields.add("npairs_result");
		mRelevantFields.add("cv_brainlv_avg");
		mRelevantFields.add("zscored_brainlv_avg");
		mRelevantFields.add("resamp1");
		mRelevantFields.add("resamp2");
		mRelevantFields.add("cv_scores");
		mRelevantFields.add("cv_scores_avg");
		mRelevantFields.add("class_labels");
		mRelevantFields.add("class_names");
		mRelevantFields.add("evals");
		mRelevantFields.add("reprod_cc");
		mRelevantFields.add("prediction");
		mRelevantFields.add("priors");
		mRelevantFields.add("pp_true_class");
	}

	protected void createResultModel() {
		mNPairsResultModel = new NPairsResultModel();
		mResultModel = mNPairsResultModel;
	}

	protected void loadOtherData() {

		MLArray temp = null;
		
		// Load full-data reference canonical eigenimage data if possible
		temp = mResultInfo.get("brainlv");
		if (temp != null) {
			mResultModel.addBrainData(BrainData.FULL_DATA_STRING, ((MLDouble)temp).getArray() );
		}
		
		if (mResultInfo != null) {
			MLArray npairs_result = mResultInfo.get("npairs_result");
			if (npairs_result != null && npairs_result.isStruct()) {
				MLStructure struct_npairs_result = (MLStructure) npairs_result;
				
				// Load average canonical eigenimage data if possible
				temp = struct_npairs_result.getField("cv_brainlv_avg");
				if (temp != null) {
					double[][] avg_canonical = ((MLDouble) temp).getArray();
					mResultModel.addBrainData(BrainData.AVG_CANONICAL_STRING, avg_canonical);
				}
				
				// Load average z-scored eigenimage data if possible
				temp = struct_npairs_result.getField("zscored_brainlv_avg");
				if (temp != null) {
					double[][] avg_zscored = ((MLDouble) temp).getArray();
					mResultModel.addBrainData(BrainData.AVG_ZSCORED_STRING, avg_zscored);
				}
				
				temp = struct_npairs_result.getField("cv_scores");
				if (temp != null) {
					mNPairsResultModel.setCvScores(((MLDouble) temp).getArray());
				}
				
				temp = struct_npairs_result.getField("class_labels");
				if (temp != null) {
					mNPairsResultModel.setClassLabels(MLFuncs.getColumn(((MLDouble) temp).getArray(), 0));
				}
				
				MLCell class_names = (MLCell) struct_npairs_result.getField("class_names");
				if (class_names != null) {
					ArrayList<String> classNames = MLFuncs.MLCell1dRow2StrArrayList(class_names);
					mNPairsResultModel.setClassNames(classNames);
				}
				
				MLArray resamp1 = struct_npairs_result.getField("resamp1");
				if (resamp1 != null && resamp1.isStruct()) {
					MLStructure struct_resamp1 = (MLStructure) resamp1;
					
					// Load evals data for resamp1 if possible
					temp = struct_resamp1.getField("evals");
					if (temp != null) {
						mNPairsResultModel.setEvals1(((MLDouble) temp).getArray());
					}
					
					temp = struct_resamp1.getField("cv_scores_avg");
					if (temp != null) {
						mNPairsResultModel.setCvScoresTrain(((MLDouble) temp).getArray());
					}
				}
				
				MLArray resamp2 = struct_npairs_result.getField("resamp2");
				if (resamp2 != null && resamp2.isStruct()) {
					MLStructure struct_resamp2 = (MLStructure) resamp2;
					
					// Load evals data for resamp2 if possible
					temp = struct_resamp2.getField("evals");
					if (temp != null) {
						mNPairsResultModel.setEvals2(((MLDouble) temp).getArray());
					}
					
					temp = struct_resamp2.getField("cv_scores_avg");
					if (temp != null) {
						mNPairsResultModel.setCvScoresTest(((MLDouble) temp).getArray());
					}
				}
				
				temp = struct_npairs_result.getField("reprod_cc");
				if (temp != null) {
					mNPairsResultModel.setReprodCC(((MLDouble) temp).getArray());
				}
				
				MLArray prediction = struct_npairs_result.getField("prediction");
				if (prediction != null && prediction.isStruct()) {
					MLStructure struct_prediction = (MLStructure) prediction;
					
					MLArray priors = struct_prediction.getField("priors");
					if (priors != null && priors.isStruct()) {
						MLStructure struct_priors = (MLStructure) priors;
						
						temp = struct_priors.getField("pp_true_class");
						if (temp != null) {
							mNPairsResultModel.setPPTrueClass(((MLDouble) temp).getArray());
						}
					}
				}
				
				temp = struct_npairs_result.getField("split_test_vols");
				if (temp != null) {
					mNPairsResultModel.setSplitVols(((MLDouble) temp).getArray());
				}
				
				temp = struct_npairs_result.getField("split_obj_labels");
				if (temp == null) {
					// try loading using obsolete variable name instead
					temp = struct_npairs_result.getField("subj_labels");
				}
				if (temp != null) {
					mNPairsResultModel.setSplitObjLabels(MLFuncs.getRow(((MLDouble) temp).getArray(), 0));
				}
				temp = struct_npairs_result.getField("split_type");
				if(temp != null){
					String splitType = ((MLChar) temp).getString(0);
					mNPairsResultModel.setSplitType(splitType);
				}else{
					mNPairsResultModel.setSplitType(null);
				}
			}
		}
	}

	public NPairsResultModel getNpairsResultModel() {
		return mNPairsResultModel;
	}
}
