package pls.chrome.result.controller.command;

import pls.chrome.result.model.GeneralRepository;

/**
 * A command for changing the transparency of the labels in the single pane
 * brain viewer.
 */
public class LabelsTransparencyCommand extends ResultsCommand {
	private int mTransparency;
	
	public LabelsTransparencyCommand(GeneralRepository repository, int transparency) {
		super(repository);
		
		mTransparency = transparency;
		
		mUndoable = false;
		mInvalidates = false;
	}

	@Override
	public boolean execDo() {
		mRepository.getImagePropertiesModel().setLabelsTransparency(mTransparency);
		
		return true;
	}

	@Override
	public boolean execUndo() {
		// Transparency commands are not undoable because of how often they
		// could occur in a short time span.
		return false;
	}
}
