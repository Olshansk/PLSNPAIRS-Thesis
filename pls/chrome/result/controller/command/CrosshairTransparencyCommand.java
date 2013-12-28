package pls.chrome.result.controller.command;

import pls.chrome.result.model.GeneralRepository;

/**
 * This command sets the transparency of the cross hair in the single pane
 * brain viewer and the three pane brain viewer.
 */
public class CrosshairTransparencyCommand extends ResultsCommand {
	
	private int mTransparency;

	public CrosshairTransparencyCommand(GeneralRepository repository, int transparency) {
		super(repository);
		
		mTransparency = transparency;
		
		// Transparency commands are not undoable because of how often they
		// could occur in a short time span.
		mUndoable = false;
		mInvalidates = false;
	}

	@Override
	public boolean execDo() {
		mRepository.getImagePropertiesModel().setCrosshairTransparency(mTransparency);
		
		return true;
	}

	@Override
	public boolean execUndo() {
		// Transparency commands are not undoable.
		return false;
	}

}
