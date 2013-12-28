package pls.chrome.result.controller.command;

import pls.chrome.result.model.BrainData;
import pls.chrome.result.model.GeneralRepository;

/**
 * This command sets the brain view.
 */
public class BrainViewCommand extends ResultsCommand {

	int mBrainView;
	int mOrigBrainView;
	
	public BrainViewCommand(GeneralRepository repository, int brainView) {
		super(repository);
		
		mBrainView = brainView;
		mOrigBrainView = mRepository.getImagePropertiesModel().getBrainView();
		
		mCommandLabel = "Set View - " + BrainData.viewToString(brainView);
	}

	@Override
	public boolean execDo() {
		mRepository.getImagePropertiesModel().setBrainView(mBrainView);
		
		return true;
	}

	@Override
	public boolean execUndo() {
		mRepository.getImagePropertiesModel().setBrainView(mOrigBrainView);
		
		return true;
	}

}
