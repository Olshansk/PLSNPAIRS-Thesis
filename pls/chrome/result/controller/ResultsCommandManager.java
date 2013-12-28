package pls.chrome.result.controller;

import java.awt.Color;
import java.io.File;
import java.util.ArrayList;

import pls.chrome.result.ThreeViewPanel;
import pls.chrome.result.blvplot.BrainInfoPane;
import pls.chrome.result.blvplot.BrainLatentVariablesPlot;
import pls.chrome.result.controller.command.BackgroundImageCommand;
import pls.chrome.result.controller.command.BrainViewCommand;
import pls.chrome.result.controller.command.ColourScaleCommand;
import pls.chrome.result.controller.command.Command;
import pls.chrome.result.controller.command.CrosshairColourCommand;
import pls.chrome.result.controller.command.CrosshairTransparencyCommand;
import pls.chrome.result.controller.command.FlipVolumeCommand;
import pls.chrome.result.controller.command.GlobalColourScaleCommand;
import pls.chrome.result.controller.command.InvertLvCommand;
import pls.chrome.result.controller.command.LabelsColourCommand;
import pls.chrome.result.controller.command.LabelsTransparencyCommand;
import pls.chrome.result.controller.command.LoadFilesCommand;
import pls.chrome.result.controller.command.RotateCommand;
import pls.chrome.result.controller.command.SelectDatatypeCommand;
import pls.chrome.result.controller.command.SelectLvCommand;
import pls.chrome.result.controller.command.SelectModelCommand;
import pls.chrome.result.controller.command.SelectVoxelCommand;
import pls.chrome.result.controller.command.SliceFiltersCommand;
import pls.chrome.result.controller.command.ToggleUseCrosshairCommand;
import pls.chrome.result.controller.command.ToggleUseDescriptionLabelsCommand;
import pls.chrome.result.controller.command.ToggleUseGlobalColourScaleCommand;
import pls.chrome.result.controller.command.ToggleUseLabelsCommand;
import pls.chrome.result.controller.command.ZoomCommand;
import pls.chrome.result.model.GeneralRepository;

public class ResultsCommandManager extends CommandManager {
	private static ResultsCommandManager mSingleton = new ResultsCommandManager();
	
	public static BrainLatentVariablesPlot mPlot = null;
	public static BrainInfoPane mBrainPane = null;
	public static ThreeViewPanel mThreeView = null;
	private static GeneralRepository mRepository = null;
//	public static CoordinatesPanel mCoordinates = null;
	
	private ResultsCommandManager(){}
	
	public static void setRepository(GeneralRepository repo){
		mRepository = repo;
	}

	///////////////////////////////////////////////////////////////////////////
	// Selection Commands
	///////////////////////////////////////////////////////////////////////////
	public static void selectResultFile(String volumeName) {
		SelectModelCommand command = new SelectModelCommand(mRepository, volumeName);
		
		mSingleton.executeCommand(command);
	}
	
	public static void selectBrainData(String dataName) {
		SelectDatatypeCommand command = new SelectDatatypeCommand(mRepository, dataName);
		
		mSingleton.executeCommand(command);
	}
	
	public static void selectLv(int lvNum) {
		SelectLvCommand command = new SelectLvCommand(mRepository, lvNum);
		
		mSingleton.executeCommand(command);
	}
	
	public static void selectVoxel(int lag, int x, int y, int z) {
		SelectVoxelCommand command = new SelectVoxelCommand(mRepository, lag, x, y, z);
		
		mSingleton.executeCommand(command);
	}
	
	///////////////////////////////////////////////////////////////////////////
	// Colour Scale Commands
	///////////////////////////////////////////////////////////////////////////
	public static void setColourScale(double max, double min, double threshold) {
		ColourScaleCommand command = new ColourScaleCommand(mRepository, max, min, threshold);
		
		mSingleton.executeCommand(command);
	}
	
	public static void setGlobalColourScale(double max, double min, double threshold) {
		GlobalColourScaleCommand command = new GlobalColourScaleCommand(mRepository, max, min, threshold);
		
		mSingleton.executeCommand(command);
	}
	
	public static void setUseGlobalScale() {
		ToggleUseGlobalColourScaleCommand command = new ToggleUseGlobalColourScaleCommand(mRepository);
		
		mSingleton.executeCommand(command);
	}
	
	///////////////////////////////////////////////////////////////////////////
	// Main Brain Viewer Image Property Commands
	///////////////////////////////////////////////////////////////////////////
	public static void setBrainView(int brainView) {
		BrainViewCommand command = new BrainViewCommand(mRepository, brainView);
		
		mSingleton.executeCommand(command);
	}
	
	public static void toggleCrosshair() {
		ToggleUseCrosshairCommand command = new ToggleUseCrosshairCommand(mRepository);
		
		mSingleton.executeCommand(command);
	}
	
	public static void setCrosshairColor(Color color) {
		CrosshairColourCommand command = new CrosshairColourCommand(mRepository, color);
		
		mSingleton.executeCommand(command);
	}
	
	public static void setCrosshairTransparency(int value) {
		CrosshairTransparencyCommand command = new CrosshairTransparencyCommand(mRepository, value);
		
		mSingleton.executeCommand(command);
	}
	
	public static void toggleLabels() {
		ToggleUseLabelsCommand command = new ToggleUseLabelsCommand(mRepository);
		
		mSingleton.executeCommand(command);
	}
	
	public static void setLabelColor(Color color) {
		LabelsColourCommand command = new LabelsColourCommand(mRepository, color);
		
		mSingleton.executeCommand(command);
	}
	
	public static void setLabelTransparency(int value) {
		LabelsTransparencyCommand command = new LabelsTransparencyCommand(mRepository, value);
		
		mSingleton.executeCommand(command);
	}
	
	public static void toggleDescriptionLabels(boolean useLabels) {
		ToggleUseDescriptionLabelsCommand command = new ToggleUseDescriptionLabelsCommand(mRepository);
		
		mSingleton.executeCommand(command);
	}
	
	public static void setImageRotation(int rotation) {
		RotateCommand command = new RotateCommand(mRepository, rotation);
		
		mSingleton.executeCommand(command);
	}
	
	public static void setImageZoom(double zoomScale) {
		ZoomCommand command = new ZoomCommand(mRepository, zoomScale);
		
		mSingleton.executeCommand(command);
	}
	
	public static void setAnatomicalLib(boolean usingNifti){
		mRepository.setAnatomicalLib(usingNifti);
	}
	///////////////////////////////////////////////////////////////////////////
	// Main Brain Image Viewer Filter Commands
	///////////////////////////////////////////////////////////////////////////
	
	public static void setFilters(int brainView, ArrayList<Integer> lagNumbers,
			ArrayList<Integer> sliceNumbers, int numRowsPerLag, boolean displayAll) {
		
		SliceFiltersCommand command = new
				SliceFiltersCommand(mRepository, brainView, lagNumbers,
				sliceNumbers, numRowsPerLag, displayAll);
		
		mSingleton.executeCommand(command);
	}
	
	public static void setViewedLvs(String dataType, ArrayList<Integer> viewedLvs) {
		mRepository.getGeneral().getViewModel().setViewedLvs(dataType, viewedLvs);
		
		mRepository.calculateColourScale();
	}
	
	public static void toggleFilterEnabled(String filterName) {
		mRepository.toggleBrainFilterEnabled(filterName);
	}
	
	public static void saveBrainSliceImages() {
		mBrainPane.doSaveAs();
	}
	
	public static void loadFiles(ArrayList<String> filenames) {
		LoadFilesCommand command = new LoadFilesCommand(mRepository, filenames);
		
		mSingleton.executeCommand(command);
	}

	
	public static void toggleInverted() {
		InvertLvCommand command = new InvertLvCommand(mRepository);
		
		mSingleton.executeCommand(command);
	}
	
	public static void flipVolume(int brainView, boolean horizontal, boolean vertical) {
		FlipVolumeCommand command = new FlipVolumeCommand(mRepository, horizontal, vertical, brainView);
		
		mSingleton.executeCommand(command);
	}
	
	public static void setLeftPanelVisible(boolean isVisible) {
		mPlot.setLeftSidePanelVisible(isVisible);
	}
	
	public static void setBackgroundImage(File path) {
		BackgroundImageCommand command;
		command = new BackgroundImageCommand(mRepository,path); 
				
		mSingleton.executeCommand(command);
	}
	
	public static boolean isUndoAvailable() {
		return mSingleton.canUndo();
	}
	
	public static void undo() {
		mSingleton.undoCommand();
	}
	
	public static Command getNextUndo() {
		return mSingleton.getNextUndoCommand();
	}
	
	public static boolean isRedoAvailable() {
		return mSingleton.canRedo();
	}
	
	public static void redo() {
		mSingleton.redoCommand();
	}
	
	public static Command getNextRedo() {
		return mSingleton.getNextRedoCommand();
	}
	
	public static Publisher getPublisher() {
		return mSingleton.publisher;
	}
	
	public static void dispose() {
		mPlot = null;
		mBrainPane = null;
		mThreeView = null;
		mRepository = null;
	}
}
