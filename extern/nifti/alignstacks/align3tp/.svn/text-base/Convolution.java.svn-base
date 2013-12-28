package extern.nifti.alignstacks.align3tp;

import java.awt.image.ColorModel;
import ij.*;
import ij.process.*;

/**
 * This class convolves the pixels in an ImagePlusPlus with a 3D kernel
 * and returns a new ImagePlusPlus.  It uses identical reg and view, so
 * operations on the new ImagePlusPlus affect the old ImagePlusPlus.  It
 * does 3D convolution on short or float processors.
 *
 *
 * @author J. Anthony Parker, MD PhD <J.A.Parker@IEEE.org>
 * @version 14July2004
 *
 * @see	Align3_TP
 * @see extern.nifti.alignstacks.align3tp.Alignment
 * @see	extern.nifti.alignstacks.align3tp.ImagePlusPlus
 * @see	extern.nifti.alignstacks.align3tp.Affine
 * @see	extern.nifti.alignstacks.align3tp.Matrix
 * @see	extern.nifti.alignstacks.align3tp.Vector
 */
class Convolution {
	private int n2 = 2, n = 2*n2+1;	// n is kernel size
	private float[] kernel;
	private double oldSigma = 0.0;
	private ImagePlusPlus impp;
	private ImagePlus imp;
	private short[][] pixelsIn = null, pixels = null;
	private float[][] pixelsInF = null, pixelsF = null;
	private int w, h, size;
	private boolean fp;

	Convolution(ImagePlusPlus impp) {
		this.impp = impp;
		imp = impp.getImp();
		w = imp.getWidth();
		h = imp.getHeight();
		size = imp.getStackSize();
		if(imp.getProcessor() instanceof FloatProcessor) {
			fp = true;
			pixelsInF = (float[][])impp.getPixels();
			pixelsF = new float[size][w*h];
		} else {
			fp = false;
			pixelsIn = (short[][])impp.getPixels();
			pixels = new short[size][w*h];
		}
		return;
	}

	ImagePlusPlus gaussian(double sigma, int kSize) {
		if(kSize!=0) {	// n is odd number <= kSize
			n2 = (kSize-1)/2;
			n = 2*n2+1;
		}
		kernel = new float[n*n*n];
		float sum = 0f;
		double s = 1.0/(2.0*sigma*sigma);	// 1/2*sigma**2
		for(int i=0, m=0; i<n; i++)
			for(int j=0; j<n; j++)
				for(int k=0; k<n; k++, m++) {
					kernel[m] = (float)(Math.exp(-sq(i-n2)*s)*
										Math.exp(-sq(j-n2)*s)*
										Math.exp(-sq(k-n2)*s));
					sum += kernel [m];
				}
		for(int i=0; i<n*n*n; i++) kernel[i] /= sum;
		doConvolution();
		return makeImppOut();
	}

	private double sq(double x) {return x*x;}

	private void doConvolution() {
		for(int k=0; k<size; k++) {
			IJ.showStatus("Smoothing = "+(k+1));
			for(int j=0; j<h; j++)
				for(int i=0; i<w; i++) {
					float sum = 0f;
					for(int z=0; z<n; z++) {
						int kk = k+z-n2;
						if(kk<0) kk = 0; if(kk>=size) kk = size-1;
						for(int y=0; y<n; y++) {
							int jj = j+y-n2;
							if(jj<0) jj = 0; if(jj>=h) jj = h-1;
							int znnyn = z*n*n+y*n, jjw = jj*w;	// hoist
							for(int x=0; x<n; x++) {
								int ii = i+x-n2;
								if(ii<0) ii = 0; if(ii>=w) ii = w-1;
								if(fp)
									sum += kernel[znnyn+x]*
											pixelsInF[kk][jjw+ii];
								else
									sum += kernel[znnyn+x]*
											(pixelsIn[kk][jjw+ii]&0xffff);
							}
						}
					}
					if(fp)
						pixelsF[k][j*w+i] = sum;
					else
						pixels[k][j*w+i] = (short)sum;
				}
		}
		return;
	}

	private ImagePlusPlus makeImppOut() {
		ColorModel cm = imp.getProcessor().getColorModel();
		ImageStack stack = new ImageStack(w, h);
		for(int k=0; k<size; k++) {
			ImageProcessor ip;
			if(fp)
				ip = new FloatProcessor(w, h, pixelsF[k], cm);
			else
				ip = new ShortProcessor(w, h, pixels[k], cm);
			stack.addSlice("Slice "+Integer.toString(k+1), ip);
		}
		ImagePlus impOut = new ImagePlus("Smoothed Pixels", stack);
		impOut.setCalibration(imp.getCalibration());
		ImagePlusPlus imppOut = new ImagePlusPlus(impOut);
		imppOut.setView(impp.getView());
		imppOut.setReg(impp.getReg());
		imppOut.setInterpolate(impp.getInterpolate());
		imppOut.setWeight(impp.getWeight());
		return imppOut;
	}

}