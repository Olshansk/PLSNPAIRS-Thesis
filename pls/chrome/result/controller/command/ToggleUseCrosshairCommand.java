package pls.chrome.result.controller.command;

import pls.chrome.result.model.GeneralRepository;

/**
 * A command for turning the cross hairs on and off in the single pane brain
 * viewer and tree pane brain viewer.
 */
public class ToggleUseCrosshairCommand extends ResultsCommand {

	public ToggleUseCrosshairCommand(GeneralRepository repository) {
		super(repository);
		
		mCommandLabel = "Toggle Crosshair";
	}

	@Override
	public boolean execDo() {
		boolean currentState = mRepository.getImagePropertiesModel().isCrosshairEnabled();
		
		mRepository.getImagePropertiesModel().setCrosshairEnabled(!currentState);
		
		return true;
	}

	@Override
	public boolean execUndo() {
		return execDo();
	}

}
