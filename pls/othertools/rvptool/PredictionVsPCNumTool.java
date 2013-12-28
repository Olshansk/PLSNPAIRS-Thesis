package pls.othertools.rvptool;

import java.awt.event.ActionEvent;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Map;

import javax.swing.JOptionPane;

import pls.shared.MLFuncs;

import com.jmatio.io.MatFileFilter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLStructure;

import extern.ArrayFuncs;
import extern.NewMatFileReader;

/*
 * The name Group has been changed to Curve. For any variable with the word
 * group in it, you can safely understand that this refers to a curve.
 */
public class PredictionVsPCNumTool extends ReproducibilityVsPredictionTool {
	
	public PredictionVsPCNumTool() {
		super("Prediction VS PC # Metaplot tool", "Split object");
	}

	@Override
	public void actionPerformed(ActionEvent e){
		if (e.getActionCommand().equals("Plot") ) {
			new PPlot(mCurves, predType);
		}
		else {
			super.actionPerformed(e);
		}
	}

	@Override
	/**
	 * Save the group information held in the pvpc metaplot tool.
	 * This action happens when the user hits File->save.
	 * @param file The chosen file path.
	 * @param extDesc The description of the chosen file filter.
	 */
	protected void saveFile(File file, String extDesc) {
		
		//Append the appropriate extension
		if(extDesc.equals(".txt filter")){
			String abspath = file.getAbsolutePath();
			if(!abspath.endsWith(".txt")){
				file = new File(abspath+".txt");
			}
		}
		
		try {
			BufferedWriter bw = new BufferedWriter(new FileWriter(file) );

			for(CurveGroup group : mCurves) {
				String line = group.getLabel() + ","
						    + group.getFilename() + "," + "SO"
							+ group.getCurveUnit() + ","
							+ group.getNumbers();
				bw.write(line);
				bw.newLine();
			}

			bw.close();
		} catch (IOException e) {
			String errorString = "Something really unexpected happened " +
					"when opening file " + file.getName() + "\n";
			JOptionPane.showMessageDialog(this, errorString, "IOError",
					JOptionPane.ERROR_MESSAGE);
		}
	}
	
	/**
	 * Check whether the result file at location 'fileName' contains the 
	 * specified session. 
	 * @param unit the session number that we want to see exist in 'fileName'
	 * @param fileName the result file in question.
	 * @return true if the mat file contains the session, false otherwise. 
	 */
	/* This function is not used because checking subject labels is simply
	 * too costly. The user is warned about invalid subject labels at the 
	 * time of plotting however.*/
	/*
		private boolean validateMatFile(String label, int unit, String fileName){
		Map<String, MLArray> resultInfo = null;
		MLStructure struct_npairs_result;
		MLArray temp;
		MLArray npairs_result;
		
		double[] subjLabels = null;
				
		try {
			 resultInfo = new NewMatFileReader(fileName,
					new MatFileFilter(new String[]{"npairs_result"}))
												   .getContent();
			 
			 npairs_result = resultInfo.get("npairs_result");
			
			 if (npairs_result != null && npairs_result.isStruct()) {
				struct_npairs_result = (MLStructure) npairs_result;
			 }else{
				 return Error("Could not load " 
						 + fileName + 
						 "\n Malformed Matlab file: " +
						 "missing npairs_result struct");
			 }
			
			temp = struct_npairs_result.getField("subj_labels");
			if(temp != null){
				subjLabels = MLFuncs.getRow(((MLDouble) temp).getArray(), 0);
				int [] realSubjLabel;
				realSubjLabel = (int []) ArrayFuncs.convertArray(subjLabels,
																int.class);
				
				if(MLFuncs.unique(realSubjLabel).length < unit){
					return Error("The group " + label + " does not contain"
							+ " the subject: " + unit); //invalid subject.
				}
				return true; // valid subject
				
			}
			return Error("Could not load " 
					+ fileName + 
					"\n Malformed Matlab file: " +
					"missing subj_labels field");
			
		} catch (IOException e) {
			return Error("Critical Error while reading file: " +
					fileName); 
		}
	}*/
	
	@Override
	/**
	 * Check whether the matfile that contains the curve we want to plot 
	 * exists and also contains the session that we are trying to plot. 
	 * @param group Contains information about the curve we are trying to plot.
	 */
	protected boolean validateExistence(CurveGroup group){
		String generalFileName = group.getFilename();
		boolean error = false;
		errorString = null; //reset error string.
		
		// Check file existence.
		for(String number : group.getParsedNumbers()){
			String fileName = generalFileName.replace("$",number);
			File checkFile = new File(fileName);
			
			if(!checkFile.canRead()){
				error = true;
				Error("The file " + checkFile.getAbsolutePath() + " either" +
								" does not exist or is unreadable.");
				continue;
			}
			//return immediately since every single file in this group should
			//have the same number of sessions/subjects. so if this test fails
			//for one file it should fail for them all. The only caveat is that
			//we also stop reporting errors if just one of the files is 
			//malformed and we ignore reporting if any of the others are
			//malformed. hopefully this wont be too common a situation.
//			if(!validateMatFile(group.getLabel(), 
//					group.getCurveUnit(),fileName)) return false;
		}
		return !error;
	}
}
