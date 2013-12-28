package pls.chrome.result.controller.command;

import java.io.File;
import java.io.IOException;
import java.util.Set;

import npairs.io.NiftiIO;

import pls.chrome.result.controller.Publisher;
import pls.chrome.result.controller.ResultsCommandManager;
import pls.chrome.result.controller.observer.brainimageproperties.BackgroundImageEvent;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.ResultModel;
import pls.sessionprofile.NiftiAnalyzeImage;
import pls.shared.GlobalVariablesFunctions;
import pls.shared.MLFuncs;

/**
 * A command to set the background image for the currently selected
 * result file.
 */
public class BackgroundImageCommand extends SelectionDependentCommand {
	private File mImage;
	private File mOldImage;
	private Publisher pub;

	private boolean updateModels(File image){
		double[] bgImageData = null;
		
		if(image != null){
			boolean usingNiftiLib = mRepository.getAnatomicalLib();
			
			if(!usingNiftiLib){
				NiftiAnalyzeImage bgImage;
				try{
					bgImage = new NiftiAnalyzeImage(image.getParent(),
												image.getName());
				}catch(Exception e){
					GlobalVariablesFunctions.showErrorMessage("Background image file "
							+ image.getAbsolutePath() + " could not be loaded.");
					return false;
				}
				bgImageData = getBgImageData(null,bgImage,false);
			}else{
				bgImageData = getBgImageData(image.getAbsolutePath(),null,true);
			}
		
			if(bgImageData == null) return false; //error occured.
		}
		
		Set<String> files = mRepository.getModels();
		for(String model : files){
			mRepository.getGeneral(model).setBgImageData(bgImageData);
		}
		pub.publishEvent(new BackgroundImageEvent() );
		mRepository.setBgImagePath(image);
		return true;
	}

	public BackgroundImageCommand(GeneralRepository repository,	File path) {
		
		super(repository);

		mImage = path;
		mOldImage = mRepository.getBgImagePath();
		
		pub = mRepository.getGeneral().getPublisher();

		if(path == null){
			mCommandLabel = "Remove Background image";
		}else{
			mCommandLabel = "Set Background Image";
		}
	}

	@Override
	protected boolean postSelectionDo() {
		return updateModels(mImage);
		
	}

	@Override
	protected boolean postSelectionUndo() {
		return updateModels(mOldImage);
	}
	
	/**
	 * Function to get the data in a NiftiAnalyzeImage as an array
	 * of doubles.  Also verifies that the dimensions of the NiftiImage
	 * are the same as the currently loaded models.
	 * THIS IS A COPY OF THE getBgImageData() function in ResultMenuBar2.java
	 */
	private double[] getBgImageData(String path, NiftiAnalyzeImage bgImage, 
			boolean useNifti) {
		ResultModel model = mRepository.getGeneral();
		int[] stDims = model.getBrainData().getDimensions();
		
		int[] bgDims = null;
		double[] bgImageData = null;
		
		if(useNifti){
			try {
				bgDims = NiftiIO.getVolDims3D(path);
				bgImageData = NiftiIO.readNiftiData(path, 0);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}else{ //mindseer
			bgDims = bgImage.get4DDimensions();
			bgImageData = bgImage.getData();
		}
		

		// Compares the resolutions of the background image and brain image.
		// Note: For the background image, its resolution is stored as [x y z t]
		// while the brain image's resolution is stored as [x y t z].

		if (bgDims[0] != stDims[0] || bgDims[1] != stDims[1]
				|| bgDims[2] != stDims[3]) {
			GlobalVariablesFunctions.showErrorMessage("Background image file "
					+ " does not have a " +
					"matching resolution to the brain image.");
			return null;
		}

		bgImageData = MLFuncs.normalize(bgImageData);
		
		return bgImageData;
	}
	
}
