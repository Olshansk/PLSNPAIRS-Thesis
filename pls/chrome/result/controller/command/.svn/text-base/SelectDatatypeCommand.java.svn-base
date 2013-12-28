package pls.chrome.result.controller.command;

import pls.chrome.result.model.GeneralRepository;

/**
 * A command for setting the selected datatype of the currently selected
 * model.
 */
public class SelectDatatypeCommand extends ResultsCommand {

	private String mDatatype;
	
	public SelectDatatypeCommand(GeneralRepository repository, String datatype) {
		super(repository);
		
		// This command is not undoable and does not invalidate the
		// command stack
		mUndoable = false;
		mInvalidates = false;
		
		mDatatype = datatype;
	}

	@Override
	public boolean execDo() {
		mRepository.getGeneral().setBrainData(mDatatype);
		
		return true;
	}

	@Override
	public boolean execUndo() {
		// selection is not undoable
		return false;
	}

}
