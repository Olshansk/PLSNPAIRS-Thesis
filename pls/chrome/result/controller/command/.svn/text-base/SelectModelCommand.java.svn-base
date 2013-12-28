package pls.chrome.result.controller.command;

import pls.chrome.result.model.GeneralRepository;

/**
 * A command for setting the selected model.
 */
public class SelectModelCommand extends ResultsCommand {

	private String mModel;
	
	public SelectModelCommand(GeneralRepository repository, String model) {
		super(repository);
		
		// This command is not undoable and does not invalidate the
		// command stack
		mUndoable = false;
		mInvalidates = false;
		
		mModel = model;
	}

	@Override
	public boolean execDo() {
		mRepository.setSelectedModel(mModel);
		
		return true;
	}

	@Override
	public boolean execUndo() {
		// Selection is not undoable
		return false;
	}

}
