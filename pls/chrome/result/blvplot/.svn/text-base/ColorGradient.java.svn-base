package pls.chrome.result.blvplot;

import java.awt.Color;

public class ColorGradient {
	
	public double max,min,thresh;
	Color grey;
		
	public ColorGradient(double max, double min, double thresh) {
//		this.max = Math.max(max, Math.abs(min));
//		this.min = -this.max;
		this.max = max;
		this.min = min;
		this.thresh = thresh;
		
		grey = new Color(0.54f,0.54f,0.54f);
	}
	
	public boolean equals(ColorGradient other) {
		if (other.max == this.max &&
				other.min == this.min &&
				other.thresh == this.thresh) {
			return true;
		}
		
		return false;
	}
	
	public boolean inThreshold(double val) {
		return (Math.abs(val) < thresh);
	}
	
	public Color getColor(double val) {
		
		if (inThreshold(val)) {
			return grey;
		}
		
		int[][] pcolors;
		int[][] ncolors;
		
		if(ColorGradientScheme.activeTheme.equals(ColorGradientScheme.TEMPERATURE)){
			pcolors = pColorsHM;
			ncolors = nColorsHM;
		}
		else{
			pcolors = positiveColors;
			ncolors = negativeColors;
		}
		
		if (val >= 0) {
			
				// Minus 1 compensates for 0-based arrays
				double colorArrayIndex = (val*(pcolors.length-1) / max);
				
				// Rounded index and converted to int
				int i = (int)Math.round(colorArrayIndex);
				i = Math.min(i, pcolors.length - 1);
				if (i < 0) {
					i = 0;
				}
				
				return new Color(pcolors[i][0],
								pcolors[i][1], 
								pcolors[i][2]);
		}
		
		//Minus 1 compensates for 0-based arrays
		double colorArrayIndex = Math.abs(Math.abs(val)*(ncolors.length-1) / min);
		
		//Rounded index and converted to int
		int i = (int)Math.round(colorArrayIndex);
		i = Math.min(i, ncolors.length - 1);
		
		//Positive colors are listed low-to-high, but negative colors
		//are listed high-to-low, so this adjusts to the appropriate index
		i = (ncolors.length-1) - i;
		
		if (i > ncolors.length - 1) {
			i = ncolors.length - 1;
		}
		
		if (val == -1) return Color.white;
		
		return new Color(ncolors[i][0],ncolors[i][1], ncolors[i][2]);
		
	}
	
	
	//Temperature colour scheme.
	private static int[][] pColorsHM = {
			{0,0,0},
			{24,0,0},
			{36,0,0},
			{48,0,0},
			{60,0,0},
			{72,0,0},
			{84,0,0},
			{96,0,0},
			{108,0,0},
			{120,0,0},
			{132,0,0},
			{144,0,0},
			{156,0,0},
			{168,0,0},
			{180,16,0},
			{192,32,0},
			{204,48,0},
			{216,64,0},
			{228,80,0},
			{240,96,0},
			{252,112,0},
			{255,128,0},
			{255,144,0},
			{255,160,32},
			{255,176,64},
			{255,192,96},
			{255,208,128},
			{255,224,160},
			{255,240,192},
			{255,255,224},
			{255,255,255},
			{255,255,255}
	};
	
	//Temperature colour scheme
	private static int[][] nColorsHM = {
			{128,255,128},
		    {112,255,143},		
		    {96,255,159},
		    {80,255,175},
		    {64,255,191},
		    {48,255,207},
		    {32,255,223},
		    {16,255,239},
		    {0,255,255},
		    {0,239,255},
		    {0,223,255},
		    {0,207,255},
		    {0,191,255},
		    {0,175,255},
		    {0,159,255},
		    {0,143,255},
		    {0,128,255},
		    {0,112,255},
		    {0,96,255},
		    {0,80,255},
		    {0,64,255},
		    {0,48,255},
		    {0,32,255},
		    {0,16,255},
		    {0,0,227},
		    {0,0,199},
		    {0,0,171},
		    {0,0,143},
		    {0,0,115},
		    {0,0,87},
		    {0,0,59},
		    {0,0,31}
	};
	
	//length = 32
	private static int[][] positiveColors = {
			{143,255,112},
			{159,255,96},
			{175,255,80},
			{191,255,64},
			{207,255,48},
			{223,255,32},
			{239,255,16},
			{255,255,0},
			{255,239,0},
			{255,223,0},
			{255,207,0},
			{255,191,0},
			{255,175,0},
			{255,159,0},
			{255,143,0},
			{255,128,0},
			{255,112,0},
			{255,96,0},
			{255,80,0},
			{255,64,0},
			{255,48,0},
			{255,32,0},
			{255,16,0},
			{255,0,0},
			{239,0,0},
			{223,0,0},
			{207,0,0},
			{191,0,0},
			{175,0,0},
			{159,0,0},
			{143,0,0},
			{128,0,0}	
	};
	
	
	//length = 32
	private static int[][] negativeColors = {
		{0,0,143},
		{0,0,159},
		{0,0,175},
		{0,0,191},
		{0,0,207},
		{0,0,223},
		{0,0,239},
		{0,0,255},
		{0,16,255},
		{0,32,255},
		{0,48,255},
		{0,64,255},
		{0,80,255},
		{0,96,255},
		{0,112,255},
		{0,128,255},
		{0,143,255},
		{0,159,255},
		{0,175,255},
		{0,191,255},
		{0,207,255},
		{0,223,255},
		{0,239,255},
		{0,255,255},
		{16,255,239},
		{32,255,223},
		{48,255,207},
		{64,255,191},
		{80,255,175},
		{96,255,159},
		{112,255,143},
		{128,255,128}	
	};
	
	public String toString() {
		String output = "[ColorGradient] max: " + max;
		output += " min: " + min;
		output += " threshold: " + thresh;
		
		return output;
	}
}


