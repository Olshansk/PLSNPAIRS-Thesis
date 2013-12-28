package pls.chrome.result.controller.command;

import pls.chrome.result.model.GeneralRepository;

/**
 * This command toggles between using a global colour scale and using
 * individual colour scales for each LV.
 */
public class ToggleUseGlobalColourScaleCommand extends ResultsCommand {

	public ToggleUseGlobalColourScaleCommand(GeneralRepository repository) {
		super(repository);
		
		mCommandLabel = "Toggle global colour scale";
	}

	@Override
	public boolean execDo() {
		boolean currentState = mRepository.getUseGlobalScale();
		
		mRepository.setUseGlobalScale(!currentState);
		
		return true;
	}

	@Override
	public boolean execUndo() {
		return execDo();
	}

}
