package pls.chrome.result.controller.command;

import pls.chrome.result.model.GeneralRepository;

/**
 * A command for setting the selected voxel for the currently selected
 * lv of the currently selected datatype of the currently selected model.
 */
public class SelectVoxelCommand extends ResultsCommand {
	private int mLag;
	private int mX;
	private int mY;
	private int mZ;
	
	public SelectVoxelCommand(GeneralRepository repository, 
			int lag, int x, int y, int z) {
		super(repository);
		
		// This command is not undoable and does not invalidate the
		// command stack
		mUndoable = false;
		mInvalidates = false;

		mLag = lag;
		mX = x;
		mY = y;
		mZ = z;
	}

	@Override
	public boolean execDo() {
		mRepository.getGeneral().getSelectionModel().selectVoxel(mX, mY, mZ, mLag);
		
		return true;
	}

	@Override
	public boolean execUndo() {
		// Selection is not undoable
		return false;
	}

}
