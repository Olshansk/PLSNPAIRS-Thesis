package pls.chrome.result;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;

import edu.washington.biostr.sig.brainj3d.data.BadDataFile;

import pls.chrome.result.model.BrainData;
import pls.chrome.result.model.ImageOverlayResultModel;
import pls.sessionprofile.NiftiAnalyzeImage;
import pls.shared.MLFuncs;

public class ImageOverlayResultLoader extends ResultLoader {

	private NiftiAnalyzeImage mImage = null;
	private ImageOverlayResultModel mImageOverlayResultModel = null;
	
	public ImageOverlayResultLoader(String filename) {
		super(filename);
	}
	
	public void loadFile() throws Exception {
		File file = new File(mFilename);
		
		String parentPath = file.getParent();
		String shortName = file.getName();
		
		try{
			mImage = new NiftiAnalyzeImage(parentPath, shortName);
		}catch(IllegalArgumentException e){
			throw new IllegalArgumentException("A programming error resulted when" +
					"attempting to parse the path to the image file.");
		}catch(BadDataFile e){
			throw new BadDataFile("Specified image file was not a valid analyze/nifti file");
		}catch(MalformedURLException e){
			throw new MalformedURLException("A programming error resulted when" +
			"attempting to parse the path to the image file.");
		}catch(IOException e){
			throw new IOException("An io error occured while attempting to process the file");
		}
		
		createResultModel();
		
		loadOtherData();
	}
	
	protected void createResultModel() {
		mImageOverlayResultModel = new ImageOverlayResultModel();
		mResultModel = mImageOverlayResultModel;

		mResultModel.setFilename(mFilename);
		mResultModel.setFileDir(new File(mFilename).getParent());
		
		// Sets the resolution of the loaded image according to how resolutions of brain
		// images of regular result files are stored.
		// Note: For the loaded image, its resolution is stored as [x y z t]
		// while the resolution of a regular result file's brain image is stored as [x y t z].
		int[] dims = mImage.get4DDimensions();
		int temp = dims[3];
		dims[3] = dims[2];
		dims[2] = temp;
		
		mResultModel.setDimensions(dims);
		
		// Since this is an image, it would only have a single lag value (a window size of one).
		mResultModel.setWindowSize(1);
	}
	
	protected void addOtherRelevantFields() {
	}
	
	protected void loadOtherData() {
		
		// Retrieves the image data and converts it to a two-dimensional array
		// representing the brain data.
		double[] imageData = mImage.getData();
		imageData = MLFuncs.normalize(imageData);
		
		double[][] brainData = convertImageData(imageData);
		mResultModel.addBrainData(BrainData.IMAGE_OVERLAY_STRING, brainData);
	}
	
	private double[][] convertImageData(double[] data) {
		double[][] result = null;
		
		
		
		return result;
	}
	
}
