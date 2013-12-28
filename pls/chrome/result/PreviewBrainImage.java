package pls.chrome.result;

import java.util.ArrayList;

import pls.chrome.result.blvplot.BrainImage;
import pls.chrome.result.blvplot.ColorGradient;
import pls.chrome.result.model.BrainData;
import pls.chrome.result.model.ResultModel;

@SuppressWarnings("serial")
public class PreviewBrainImage extends BrainImage {
	
	public boolean flipHorizontal;
	public boolean flipVertical;

	public PreviewBrainImage(ResultModel model,
			ArrayList<Integer> coords, String file, String type, int lvNum,
			int lagNum, int sliceNum, int brainView, ColorGradient colGrad) {
		super(model, coords, file, type, lvNum, lagNum, sliceNum,
				brainView, colGrad, 3.0, 0, false,
				0, null, false, 0, null);
		
	}
	
	protected double getValue(int coord) {
		BrainData bData = mResultModel.getBrainData(type);
		
		int[] viewCoord = bData.convert1DtoView(coord, brainView);
		
		int width = bData.getWidth(brainView);
		int height = bData.getHeight(brainView);
		
		int x = viewCoord[0];
		int y = viewCoord[1];
		
		if (flipHorizontal) {
			x = width - x;
		}
		
		if (flipVertical) {
			y = height - y;
		}
		
		int newCoord = bData.convertViewto1D(x, y, viewCoord[2], brainView);
		
		return mResultModel.getBrainData(type).getValue1D(newCoord, lagNum, lvNum);
	}
}
