package pls.chrome.result.controller.command;

import pls.chrome.result.model.BrainData;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.ResultModel;

/**
 * This sub-class of commands represents commands that require a particular
 * file, type lv or voxel be selected prior to execution.  Commands of this
 * type will record the selection state when they are created and recreate this
 * state during its do/undo.
 */
public abstract class SelectionDependentCommand extends ResultsCommand {
	SelectModelCommand mSelectModelCommand;
	SelectDatatypeCommand mSelectDatatypeCommand;
	SelectLvCommand mSelectLvCommand;
	SelectVoxelCommand mSelectVoxelCommand;
	
	public SelectionDependentCommand(GeneralRepository repository) {
		super(repository);
		
		// Create some selection commands so that undo/redo will affect the
		// correct lv, type and file.
		mSelectModelCommand = new SelectModelCommand(mRepository, mRepository.getSelectedResultFile() );
		ResultModel model = mRepository.getGeneral();
		mSelectDatatypeCommand = new SelectDatatypeCommand(mRepository, model.getSelectedDataType() );
		BrainData brainData = model.getBrainData();
		mSelectLvCommand = new SelectLvCommand(mRepository, brainData.getLv() );
		int[] selectedVoxel = model.getSelectionModel().getSelectedVoxel();
		mSelectVoxelCommand = new SelectVoxelCommand(mRepository, selectedVoxel[3],
				selectedVoxel[0], selectedVoxel[1], selectedVoxel[2]);
	}

	@Override
	public boolean execDo() {
		restoreSelectionState();
		return postSelectionDo();
	}
	
	protected abstract boolean postSelectionDo();

	@Override
	public boolean execUndo() {
		restoreSelectionState();
		return postSelectionUndo();
	}
	
	protected abstract boolean postSelectionUndo();

	/**
	 * Restores the selection state to the state saved during the creation
	 * of the command.
	 */
	private void restoreSelectionState() {
		mSelectModelCommand.execDo();
		mSelectDatatypeCommand.execDo();
		mSelectLvCommand.execDo();
		mSelectVoxelCommand.execDo();
	}
}
