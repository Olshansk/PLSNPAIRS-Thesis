package pls.chrome.result.controller.command;

import pls.chrome.result.model.GeneralRepository;

/**
 * A command for setting the selected lv of the currently selected datatype
 * of the currently selected model.
 */
public class SelectLvCommand extends ResultsCommand {

	int mLv;
	
	public SelectLvCommand(GeneralRepository repository, int lv) {
		super(repository);
		
		// This command is not undoable and does not invalidate the
		// command stack
		mUndoable = false;
		mInvalidates = false;
		
		mLv = lv;
	}

	@Override
	public boolean execDo() {
		mRepository.getGeneral().getBrainData().setLv(mLv);
		
		return true;
	}

	@Override
	public boolean execUndo() {
		// selection is not undoable
		return false;
	}

}
