package extern.nifti.alignstacks.align3tp;

import ij.gui.GenericDialog;
import ij.ImagePlus;
import ij.process.ImageProcessor;

/**
 * This class, which displays ip in windows, is used for debugging.
 *
 * @author J. Anthony Parker, MD PhD <J.A.Parker@IEEE.org>
 * @version 22January2003
 */
class ShowSlices {
	ImageProcessor[] ip;
	int len;

	ShowSlices(ImageProcessor ip) {
		this.ip = new ImageProcessor[1];
		this.ip[0] = ip;
		len = 1;
		return;
	}

	ShowSlices(ImageProcessor[] ip) {
		this.ip = ip;
		len = ip.length;
		return;
	}

	void show() {
		show("Show ImageProcessor(s)");
		return;
	}

	void show(String q) {
		String[] c = {"Skip", q};
		GenericDialog gd = new GenericDialog("Debugging");
		gd.addChoice("Action: ", c, c[0]);
		gd.showDialog();
		if(gd.wasCanceled()) return;
		switch (gd.getNextChoiceIndex()) {
			case 1:
				break;
			case 0:
			default:
				return;
		}
		for(int i=0; i<len; i++) {
			ImagePlus imp = new ImagePlus("Image "+(i+1), ip[i]);
			imp.show();
		}
		throw new RuntimeException("Exit from ShowSlices");
	}

}