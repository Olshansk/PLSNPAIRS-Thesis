package pls.chrome.result.controller.command;

import java.util.ArrayList;

import pls.chrome.result.model.GeneralRepository;

public class LoadFilesCommand extends ResultsCommand {

	private ArrayList<String> mFilenames;
	
	public LoadFilesCommand(GeneralRepository repository, ArrayList<String> filenames) {
		super(repository);
		
		mFilenames = filenames;
		
		// You cannot undo opening files, and opening files invalidates
		// the undo stack.
		mUndoable = false;
		mInvalidates = true;
	}

	@Override
	public boolean execDo() {
		return mRepository.setLoadedFiles(mFilenames);
	}

	@Override
	public boolean execUndo() {
		// Opening files is not undoable
		return false;
	}
}
