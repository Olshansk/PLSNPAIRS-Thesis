package pls.chrome.result.controller.command;

import java.util.ArrayList;

import pls.chrome.result.model.BrainData;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.ResultModel;
import pls.chrome.result.model.NPairsResultModel;
import pls.chrome.result.model.PlsResultModel;
//This command also corresponds to the invert CV command.
public class InvertLvCommand extends SelectionDependentCommand {
	public InvertLvCommand(GeneralRepository repository) {
		super(repository);
		
		mCommandLabel = "Invert LV";
	}

	@Override
	protected boolean postSelectionDo() {
                //get the currently selected result file and lv.
                ResultModel model = mRepository.getGeneral();
                int lv = mRepository.getGeneral().getBrainData().getLv();
                ArrayList<String> dataNames = mRepository.getGeneral().getDataNames();

                //We are inverting a NPairsResult file, lv and cv are
                //interchangable terms here. we invert cv values.
                if(model instanceof NPairsResultModel){
                    NPairsResultModel nPairsModel = (NPairsResultModel) model;
                    double[][] cvScores = nPairsModel.getCvScores();
                    double[][] cvTest = nPairsModel.getCvScoresTest();
                    double[][] cvTrain = nPairsModel.getCvScoresTrain();

                    for(int i = 0; i< cvScores.length; i++)
                        cvScores[i][lv] = -cvScores[i][lv];

                    for (int i = 0; i < cvTest.length; i++)
                        cvTest[i][lv] = -cvTest[i][lv];

                    for(int i = 0; i < cvTrain.length; i++)
                        cvTrain[i][lv] = -cvTrain[i][lv];

                }
                else if(model instanceof PlsResultModel){
                                        
                    //get the associated brain,design,design lv scores and invert them  
                    double[][] bScr = model.getBrainScores();
                    double[][] dScr = model.getDesignScores();
                    double[][] dlvScr = model.getDesignLv();
                    
                    for(int i = 0; i < bScr.length; i++)
                        bScr[i][lv] = -bScr[i][lv];
                    
                    for(int i = 0; i < dScr.length; i++)
                        dScr[i][lv] = -dScr[i][lv];
                
                    for(int i = 0; i < dlvScr.length; i++)
                        dlvScr[i][lv] = -dlvScr[i][lv];
                }else{
                    return false; //assume something went bad.
                }

		//for each type of data i.e Average Z-scored eigenvalue retrieve the
		//corresponding braindata and invert the lv for it.
		for (int i = 0; i < dataNames.size(); i++) {
			BrainData b = mRepository.getGeneral().getBrainData(dataNames.get(i));

			//BrainData will notify all Observers (Plots etc) of changes at invertLv().
			b.invertLv(lv);
			mRepository.calculateColourScale();
		}
		return true;
	}

	@Override
	protected boolean postSelectionUndo() {
		return postSelectionDo();
	}
}
