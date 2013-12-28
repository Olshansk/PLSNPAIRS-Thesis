package pls.chrome.result.controller.command;

import pls.chrome.result.model.GeneralRepository;

public class ToggleUseLabelsCommand extends ResultsCommand {

	public ToggleUseLabelsCommand(GeneralRepository repository) {
		super(repository);
		
		mCommandLabel = "Toggle Labels";
	}

	@Override
	public boolean execDo() {
		boolean currentState = mRepository.getImagePropertiesModel().labelsEnabled();
		
		mRepository.getImagePropertiesModel().setLabelsEnabled(!currentState);
		
		return true;
	}

	@Override
	public boolean execUndo() {
		return execDo();
	}

}
