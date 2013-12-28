package pls.chrome.result.controller.command;

import pls.chrome.result.model.GeneralRepository;

/**
 * A command to toggle the use of description labels.  Description labels
 * are the labels that indicate the ResultFile, Datatype and LV in the
 * single pane brain viewer.
 */
public class ToggleUseDescriptionLabelsCommand extends ResultsCommand {

	public ToggleUseDescriptionLabelsCommand(GeneralRepository repository) {
		super(repository);
		
		mCommandLabel = "Toggle Description Labels";
	}

	@Override
	public boolean execDo() {
		boolean currentState = mRepository.getImagePropertiesModel().descriptionLabelsEnabled();
		
		mRepository.getImagePropertiesModel().setDescriptionLabelsEnabled(!currentState);
		
		return true;
	}

	@Override
	public boolean execUndo() {
		return execDo();
	}

}
