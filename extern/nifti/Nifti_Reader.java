package extern.nifti;
import java.io.*; 
import java.awt.*; 
import ij.*; 
import ij.plugin.*;
import ij.process.*;
import ij.io.*;
import ij.measure.*;

/** This plugin loads Analyze and Nifti-1 format files. 
    	If the <filename> provided (or selected) ends with ".hdr" or ".img"
	then <filename>.hdr will be parsed and used to appropriately load 
	the raw image data found in '<filename>.img'. Otherwise the file is
	assumed to be in the concatenated nifti format. 
	If the header information conforms to the nifti-1 standard, the 
	information is stored in property "nifti" as an object of class 
	NiftiHeader.

      - Loads either big or little endian format.   

    Guy Williams, gbw1000@wbic.cam.ac.uk        19/08/2005 
*/


public class Nifti_Reader extends ImagePlus implements PlugIn {
  
	private boolean littleEndian = false;
	private boolean isNiftiData = false;
  
	private double cal_min = 0.0; 
	private double cal_max = 0.0; 

	private int nChannels = 1;
	private int depth = 1;
	private int frames = 1;

	public NiftiHeader nfti_hdr;

	public void run(String arg) {
		OpenDialog od = new OpenDialog("Open Nifti...", arg);
		String directory = od.getDirectory();
		String name = od.getFileName();
		if (name==null) return;
		IJ.showStatus("Opening: " + directory + name);
		ImagePlus imp = load(directory, name);
		if (imp!=null) {
			setStack( imp.getTitle(), imp.getStack() );	
			
			Calibration c = imp.getCalibration(); 
			boolean isSigned16Bit = c.isSigned16Bit();
			if (nfti_hdr != null) { 
				double [] coeff = new double[2];
				coeff[0] = nfti_hdr.scl_inter;
				coeff[1] = nfti_hdr.scl_slope;
				if (coeff[1] == 0.0) coeff[1] = 1.0; // If zero slope, assume unit slope
				if (isSigned16Bit) coeff[0] -= 32768.0 * coeff[1];
				c.setFunction(Calibration.STRAIGHT_LINE, coeff, "gray value");
				cal_max = (cal_max - coeff[0]) / coeff[1];
				cal_min = (cal_min - coeff[0]) / coeff[1];
			} else { 
				if (isSigned16Bit) { 
					cal_max += 32768.0;	
					cal_min += 32768.0;
				}
			}
			if (cal_max != cal_min) getProcessor().setMinAndMax(cal_min, cal_max-1.0);
			ImageStack stack = getStack();
			if (!isNiftiData) { 
				for (int i=1; i<=stack.getSize(); i++) {
					ImageProcessor ip = stack.getProcessor(i);
					ip.flipVertical();
				}
			} else { 
				CoordinateMapper [] mp = getCoors(nfti_hdr);
				if (mp!=null) setProperty("coors", mp );	
				setProperty("nifti", nfti_hdr);	
			}
			setCalibration(c);	
			/* If we ran out of memory, cut the number of slices
			 * to keep the channel/depth/frame settings consistent */
			if (nChannels*depth*frames != stack.getSize()) { 
				int oldSize = stack.getSize();
				nChannels = oldSize / (depth * frames);
				if (nChannels==0) {
					nChannels = 1;
					frames = oldSize / depth; 
					if (frames==0) { 
						frames = 1;
						depth = oldSize;
					}
				}
				for (int i=nChannels*depth*frames; i<oldSize; i++) { 
					stack.deleteLastSlice();
				}
			}
			setDimensions(nChannels, depth, frames);
			if (nChannels!=1) reshuffleStack( stack.getImageArray(), depth*frames, stack.getSize() ); 
			if (arg.equals("")) show();
		}
	}

	public ImagePlus load(String directory, String name) {
    
		FileInfo fi = new FileInfo(); 
   		String hdrName = name; 
		String imgName = name; 

		if ((name == null) || (name == "")) return null;
		if ((name.endsWith(".img")) || (name.endsWith(".hdr"))) { 
			name = name.substring(0, name.length()-4 ); 
			hdrName = name+".hdr"; 
			imgName = name+".img";
		} else { 
			hdrName = name; 
			imgName = name; 
		}

		if (!directory.endsWith(File.separator)) directory += File.separator;

		IJ.showStatus("Reading Header File: " + directory + hdrName);
    
		try {
			fi = readHeader( directory+hdrName );
			if (fi==null) return null; 	
		} catch (IOException e) { 
			IJ.log("FileLoader: "+ e.getMessage()); 
		}
		if (isNiftiData) { 
			IJ.showStatus("Reading Nifti File: " + directory + imgName ); 
		} else { 
			IJ.showStatus("Reading Analyze File: " + directory + imgName ); 
		}
		fi.fileName = imgName;
		fi.directory = directory;
		fi.fileFormat = fi.RAW;
		FileOpener fo = new FileOpener(fi);  
		ImagePlus imp = fo.open(false);
		return imp; 
	} 
 
	public FileInfo readHeader( String hdrfile ) throws IOException {

		FileInputStream filein = new FileInputStream (hdrfile);
		DataInputStream input = new DataInputStream (filein);
		FileInfo fi = new FileInfo();
		byte[] units = new byte[4]; 

		this.littleEndian = false;     

		int i;
		short bitsallocated, datatype;

//  header_key  

		input.readInt (); 				// sizeof_hdr
		for (i=0; i<10; i++) input.readByte();		// data_type
		for (i=0; i<18; i++) input.readByte(); 		// db_name 
		input.readInt (); 				// extents 
		input.readShort (); 				// session_error
		input.readByte ();				// regular 
		byte dim_info = input.readByte (); 				// hkey_un0 

// image_dimension
			
		short [] dim = new short [8];
		dim[0] = readShort (input);		// dim[0] 
		if ((dim[0] < 0) || (dim[0] > 7)) { 
			littleEndian = true;
			fi.intelByteOrder = true; 
			dim[0] = (short) (dim[0] >> 8);	
		}  
		for (i=1;i<8; i++) dim[i] = readShort(input); 
		fi.width = dim[1];				// dim[1] 
		fi.height = dim[2];				// dim[2] 
		int nImages = 1; 
		for (i=3; i<=dim[0]; i++) nImages *= dim[i]; 
		fi.nImages = nImages; 				// dim[3-7] 
		
		input.read (units, 0, 4); 			// vox_units 
		float intent_p1; 
		if (this.littleEndian) { 
			intent_p1 = Float.intBitsToFloat( (units[3]&0xff)<<24 | 
					(units[2]&0xff)<<16 | (units[1]&0xff)<<8 | (units[0]&0xff) ); 

		} else {
			intent_p1 = Float.intBitsToFloat( (units[0]&0xff)<<24 | 
					(units[1]&0xff)<<16 | (units[2]&0xff)<<8 | (units[3]&0xff) ); 
		}
		fi.unit = (new String (units, 0, 4)).trim(); 
		float intent_p2 = readFloat(input); 		// cal_units[0-3] or intent_p2 
		float intent_p3 = readFloat(input); 		// cal_units[4-7] or intent_p3  
		short intent_code = readShort(input);		// unused1 or intent_code
		datatype = readShort( input );			// datatype 
		bitsallocated = readShort( input );		// bitpix
		short slice_start = readShort (input);		// dim_un0 or slice_start
		float [] pixdim = new float[8];
		for (i=0; i<8; i++) pixdim[i] = readFloat(input);	// pixdim[0-7]  
		fi.pixelWidth = (double) pixdim[1];	 
		fi.pixelHeight = (double) pixdim[2];  
		fi.pixelDepth = (double) pixdim[3]; 
		fi.frameInterval = (double) pixdim[4]; 
		fi.offset = (int) readFloat(input);		// vox_offset
		float scl_slope = readFloat (input);		// roi_scale		or scl_slope 
		float scl_inter = readFloat (input);		// funused1 		or scl_inter 
		short slice_end = readShort(input);		// funused2		or slice_end
   		byte slice_code = input.readByte(); 		// 			& slice_code
		byte xyzt_units = input.readByte();			//			xyzt_units	
		cal_max = readFloat (input);			// cal_max 
		cal_min = readFloat (input);			// cal_min 
		float slice_duration = readFloat (input);	// compressed 		or slice_duration
		float toffset = readFloat (input);			// verified 		or toffset
		
    //   ImageStatistics s = imp.getStatistics();
		readInt (input);	//(int) s.max		// glmax 
		readInt (input);	//(int) s.min		// glmin 

// data_history 

		byte [] descBuf = new byte[80]; 
		for (i=0; i<80; i++) descBuf[i] = input.readByte();
		String descrip = new String(descBuf); 		// descrip	
		byte [] auxBuf = new byte[24]; 
		for (i=0; i<24; i++) auxBuf[i] = input.readByte();
		String aux_file = new String(auxBuf); 		// aux_file	
   
		short qform_code = readShort(input); 		//			qform_code
		short sform_code = readShort(input); 		//			sform_code
		
		float quatern_b = readFloat(input); 		// 			quatern_b
		float quatern_c = readFloat(input); 		// 			quatern_c
		float quatern_d = readFloat(input); 		// 			quatern_d
		float qoffset_x = readFloat(input); 		// 			qoffset_x
		float qoffset_y = readFloat(input); 		// 			qoffset_y 
		float qoffset_z = readFloat(input); 		// 			qoofset_z
		
		float [] srow_x = new float[4]; 
		float [] srow_y = new float[4]; 
		float [] srow_z = new float[4]; 
		for (i=0;i<4;i++) srow_x[i] = readFloat(input);	//			srow_x
		for (i=0;i<4;i++) srow_y[i] = readFloat(input); //			srow_y 
		for (i=0;i<4;i++) srow_z[i] = readFloat(input);	//			srow_z
		
		byte [] intentBuf = new byte[16];
		for (i=0; i<16; i++) intentBuf[i] = input.readByte();
		String intent_name = new String( intentBuf ); 	// 			intent_name

		byte [] magicBuf = new byte[4]; 
		for (i=0; i<4; i++) magicBuf[i] = input.readByte(); //			magic
		String magic = new String(magicBuf,0,3);
		if ((magicBuf[3]==0) && (magic.equals("ni1") || magic.equals("n+1"))) { 
			isNiftiData = true; 
			nfti_hdr = new NiftiHeader(); 
			nfti_hdr.dim_info = dim_info; 
			nfti_hdr.dim = dim; 
			nfti_hdr.intent_p1 = intent_p1;
			nfti_hdr.intent_p2 = intent_p2;
			nfti_hdr.intent_p3 = intent_p3;
			nfti_hdr.intent_code = intent_code;
			nfti_hdr.datatype = datatype;
			nfti_hdr.bitpix = bitsallocated;
			nfti_hdr.slice_start = slice_start;
			nfti_hdr.pixdim = pixdim;
			nfti_hdr.vox_offset = (float) fi.offset;
			nfti_hdr.scl_slope = scl_slope;
			nfti_hdr.scl_inter = scl_inter;
			nfti_hdr.slice_end = slice_end;
			nfti_hdr.slice_code = slice_code;
			nfti_hdr.xyzt_units = xyzt_units;
			nfti_hdr.cal_max = (float) cal_max;
			nfti_hdr.cal_min =  (float) cal_min;
			nfti_hdr.slice_duration = slice_duration;
			nfti_hdr.toffset = toffset;
			nfti_hdr.glmax = 0;
			nfti_hdr.glmin = 0;
			nfti_hdr.descrip = descrip;
			nfti_hdr.aux_file = aux_file;
			nfti_hdr.qform_code = qform_code;
			nfti_hdr.sform_code = sform_code;
			nfti_hdr.quatern_b = quatern_b;
			nfti_hdr.quatern_c = quatern_c;
			nfti_hdr.quatern_d = quatern_d;
			nfti_hdr.qoffset_x = qoffset_x;
			nfti_hdr.qoffset_y = qoffset_y;
			nfti_hdr.qoffset_z = qoffset_z;
			nfti_hdr.srow_x = srow_x;
			nfti_hdr.srow_y = srow_y;
			nfti_hdr.srow_z = srow_z;
			nfti_hdr.intent_name = intent_name;

		} else { 
			isNiftiData = false; 
		}
		
		input.close();
		filein.close();
    
		switch (datatype) {
      
			case NiftiHeader.DT_UNSIGNED_CHAR:
				fi.fileType = FileInfo.GRAY8; 			// DT_UNSIGNED_CHAR 
				bitsallocated = 8;
				break;
			case NiftiHeader.DT_SIGNED_SHORT:
				fi.fileType = FileInfo.GRAY16_SIGNED; 		// DT_SIGNED_SHORT 
				bitsallocated = 16;
				break;
			case NiftiHeader.DT_SIGNED_INT:
				fi.fileType = FileInfo.GRAY32_INT; 		// DT_SIGNED_INT
				bitsallocated = 32;
				break; 
			case NiftiHeader.DT_FLOAT:
				fi.fileType = FileInfo.GRAY32_FLOAT; 		// DT_FLOAT 
				bitsallocated = 32;
				break; 
			case NiftiHeader.DT_RGB:
				fi.fileType = FileInfo.RGB_PLANAR; 		// DT_RGB
				bitsallocated = 24; 
				break; 
			case NiftiHeader.DT_UINT16:
				fi.fileType = FileInfo.GRAY16_UNSIGNED;
				bitsallocated = 16;
				break;
			default:
				IJ.log("Data type " + datatype + "not supported\n"); 
				return null;	
		}
		if ((dim[0] > 5) && (dim[3]*dim[4]*dim[5] != fi.nImages)) { 
			IJ.log(dim[0]+"-D data not supported\n");
		} else { 
			depth = (dim[0]<3) ? 1 : dim[3];
			frames = (dim[0]<4) ? 1 : dim[4];
			nChannels = (dim[0]<5) ? 1 : dim[5];	
		}
		
		if (isNiftiData) { 
			int xyz_units = xyzt_units & 7; 
			if (xyz_units == NiftiHeader.UNITS_METER ) { 
				fi.unit = "m";
			} else if (xyz_units == NiftiHeader.UNITS_MM ) {
				fi.unit = "mm";
			} else if (xyz_units == NiftiHeader.UNITS_MICRON ) {
				fi.unit = "um";
			}
			int t_units = xyzt_units & 24; 
			if (t_units == NiftiHeader.UNITS_MSEC ) { 
				fi.frameInterval *= 0.001;
			} else if (t_units ==  NiftiHeader.UNITS_USEC ) { 
				fi.frameInterval *= 0.000001; 
			}
			
		}

		return (fi);
	}

	/* Assume the 5th dimension is "channels" */
	private void reshuffleStack(Object [] stack, int gap, int length) { 
		Object [] oldStack = new Object[ stack.length ];
		for (int i=0; i<oldStack.length; i++) oldStack[i] = stack[i];

		for (int i=0, n=0; i<gap; i++) { 
			for (int c=i; c<length; c+=gap, n++) {
				stack[n] = oldStack[c];
			}
		}
	}
	
	private CoordinateMapper[] getCoors( NiftiHeader nfti_hdr ) { 
		CoordinateMapper qmapper=null, smapper=null; 
		if (nfti_hdr.qform_code != NiftiHeader.NIFTI_XFORM_UNKNOWN ) { 
			double [] q = new double[5]; 
			q[0] = nfti_hdr.pixdim[0];
			q[2] = nfti_hdr.quatern_b; 
			q[3] = nfti_hdr.quatern_c; 
			q[4] = nfti_hdr.quatern_d; 
			double [] offset = new double[3];
			offset[0] = nfti_hdr.qoffset_x;
			offset[1] = nfti_hdr.qoffset_y;
			offset[2] = nfti_hdr.qoffset_z;
			double [] pixdim = new double[] { nfti_hdr.pixdim[1],  nfti_hdr.pixdim[2],  nfti_hdr.pixdim[3] };
			qmapper = new QuaternCoors( q, pixdim, offset, 
				CoordinateMapper.NIFTI, NiftiHeader.getCoorTypeString( nfti_hdr.qform_code ) ); 
		}
		if (nfti_hdr.sform_code != NiftiHeader.NIFTI_XFORM_UNKNOWN ) {
			double [][] m = new double[3][4]; 
			for (int i=0; i<4; i++) { 
				m[0][i] = nfti_hdr.srow_x[i];
				m[1][i] = nfti_hdr.srow_y[i];
				m[2][i] = nfti_hdr.srow_z[i];
			}
			smapper = new AffineCoors( m, CoordinateMapper.NIFTI,
					NiftiHeader.getCoorTypeString( nfti_hdr.sform_code ));
		}
		if ((qmapper==null) && (smapper==null)) return null;
		if ((qmapper!=null) && (smapper==null)) return new CoordinateMapper[] { qmapper };
		if ((qmapper==null) && (smapper!=null)) return new CoordinateMapper[] { smapper };
		return new CoordinateMapper[] { qmapper, smapper }; 
	}

	public int readInt(DataInputStream input) throws IOException {
		if (!littleEndian) return input.readInt(); 
		byte b1 = input.readByte();
		byte b2 = input.readByte();
		byte b3 = input.readByte();
		byte b4 = input.readByte();
		return ( (((b4 & 0xff) << 24) | ((b3 & 0xff) << 16) | ((b2 & 0xff) << 8) | (b1 & 0xff)) );
	}
  
	public short readShort(DataInputStream input) throws IOException {
		if (!littleEndian) return input.readShort(); 
		byte b1 = input.readByte();
		byte b2 = input.readByte();
		return ( (short) (((b2 & 0xff) << 8) | (b1 & 0xff)) );
	}
  
	public float readFloat(DataInputStream input) throws IOException {
		if (!littleEndian) return input.readFloat();  
		int orig = readInt(input);
		return (Float.intBitsToFloat(orig));
	}
}

