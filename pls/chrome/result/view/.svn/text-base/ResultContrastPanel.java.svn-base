package pls.chrome.result.view;

import java.awt.BorderLayout;
import java.io.File;
import java.util.ArrayList;

import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.PlsResultModel;
import pls.chrome.result.model.ResultModel;
import pls.chrome.shared.ContrastPanel;
import pls.chrome.shared.FilePathCheck;
import pls.shared.GlobalVariablesFunctions;
import pls.shared.MLFuncs;

@SuppressWarnings("serial")
public class ResultContrastPanel extends AbstractPlot {
	
	private ContrastPanel currentPanel;
	private FilePathCheck filePathCheck;
	
	public ResultContrastPanel(String title, GeneralRepository repository) {
		super(title, repository, GlobalVariablesFunctions.PLS);
		
		remove(mChartPanel);
	}
	
	@Override
	public void makeChart(int fileIndex) {
		if (currentPanel != null) {
			remove(currentPanel);
		}

		ResultModel model = mRepository.getGeneral(mResultFilePaths.get(fileIndex) );
		double[][] data = mRangeData.get(fileIndex);
		ArrayList<String> conditions = model.getConditionNames();
		
		currentPanel = new ContrastPanel(conditions, 
				model.getSessionProfileArray().size(),this);

		currentPanel.fillTable(data);
		add(currentPanel, BorderLayout.CENTER);
		
		if (getParent() != null) {
			getParent().repaint();
		}
	}
	
	@Override
	public double[][] getRangeData(ResultModel model) {
		if (filePathCheck == null) {
			filePathCheck = new FilePathCheck();
		}
		
		String filePath = filePathCheck.getExistingFilePath("contrast", model.getConstrastFilename(), model.getFileDir());
		if (filePath == null) {
			return null;
		}
		
		double[][] values;
		try {
			values = MLFuncs.transpose(MLFuncs.load(filePath));
		} catch (Exception ex) {
			GlobalVariablesFunctions.showErrorMessage("Unable to read file " + filePath + ".");
			return null;
		}
		
		return values;
	}
	
	protected boolean ModelIsApplicable(ResultModel model) {
		
		String contrastFile = model.getConstrastFilename();
		
		boolean sessionEmpty = false;

		if( model.getSessionProfiles() == null)
			sessionEmpty = true;

		//If a non rotated behavior pls result file has a contrast file of 
		//"BEHAV" then it is a result file without contrast information.
		//Newer nrbpls files have the path to their contrast files in this 
		//field but old files have "BEHAV" set as this value.
		
		return contrastFile != null &&
			!contrastFile.equals("NONE") &&
			!contrastFile.equals("BEHAV") && 
			!contrastFile.equals("MULTIBLOCK") &&
			!sessionEmpty;

	}
}
