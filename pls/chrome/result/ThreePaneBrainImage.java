package pls.chrome.result;

import java.awt.Color;
import java.util.ArrayList;

import pls.chrome.result.blvplot.BrainImage;
import pls.chrome.result.blvplot.ColorGradient;
import pls.chrome.result.model.ResultModel;

@SuppressWarnings("serial")
public class ThreePaneBrainImage extends BrainImage {
	ThreePaneBrainImage(ResultModel model, ArrayList<Integer> coords, String file, int sliceNum,
			int lagNum, int brainView, ColorGradient grad, double scale, int rot90times,
			boolean useCrosshair, int crosshairTransparency, Color crosshairColor, boolean useLabel, int labelTransparency, Color labelColor) {
		super(model, coords, file, model.getSelectedDataType(), model.getBrainData().getLv(), lagNum, sliceNum, brainView, grad, scale,
				rot90times, useCrosshair, crosshairTransparency, crosshairColor, useLabel, labelTransparency, labelColor);

		mDrawOutlineBox = false;
	}
}