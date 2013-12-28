package pls.chrome.result;

import java.awt.Frame;
import java.awt.Toolkit;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Map;

import javax.swing.JDialog;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JProgressBar;
import javax.swing.SpringLayout;

import com.jmatio.io.MatFileFilter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;

import extern.NewMatFileReader;

import pls.chrome.result.model.BrainData;
import pls.chrome.result.model.PlsResultModel;
import pls.shared.BfMRIVoxelDataFileFilter;
import pls.shared.GlobalVariablesFunctions;
import pls.shared.MLFuncs;
import pls.shared.VoxelLocationFileFilter;
import pls.shared.fMRIVoxelDataFileFilter;

public class MultipleVoxelExtraction {
	
	private Frame mParent;
	private String mDirectory;
	private String mFilePrefix;
	private String mFileExtension;
	
	private int mNumLags;
	private int[] mOrigin;
	private double[] mVoxelSize;
	private int[] mDimensions;
	private ArrayList<Integer> mCoordinates;
	private BrainData mCurrentBrainData;
	private ArrayList<ArrayList<String>> mDatamatProfiles;
	
	private boolean mFileLoaded;
	private ArrayList<int[]> mVoxelLocations;
	private int mNeighborhoodSize;
	
	private JProgressBar mProgressBar;

	public MultipleVoxelExtraction(Frame frame, String directory, PlsResultModel model) {
		mFileLoaded = false;
		mVoxelLocations = new ArrayList<int[]>();
		mNeighborhoodSize = 0;
		
		mProgressBar = new JProgressBar(JProgressBar.HORIZONTAL);
		
		mParent = frame;
		mDirectory = directory;
		
		mDatamatProfiles = model.getDatamatProfiles();
		
		mNumLags = model.getWindowSize();
		
		mCurrentBrainData = model.getBrainData();
		mOrigin = mCurrentBrainData.getOrigin();
		mVoxelSize = mCurrentBrainData.getVoxelSize();
		mCoordinates = new ArrayList<Integer>(model.getFilteredCoordinates() );
		
		// Retrieves only the x, y and z dimension values.
		int[] dimensions = mCurrentBrainData.getDimensions();
		mDimensions = new int[]{dimensions[0], dimensions[1], dimensions[3]};
		
		JFileChooser chooser = new JFileChooser(mDirectory);
		chooser.setFileFilter(new VoxelLocationFileFilter());
		
		// Prompts the user for the voxel location file and attempts to
		// load it.
		int option = chooser.showDialog(mParent, "Load Voxel Location File");
		if (option == JFileChooser.APPROVE_OPTION) {
			File file = chooser.getSelectedFile();
			mDirectory = file.getParent();
			
			try {
				BufferedReader br = new BufferedReader(new FileReader(file.getAbsolutePath()));
				String line = br.readLine();
				while (line != null) {
					if (!areVoxelCoordsValid(file.getName(), line)) {
						return;
					}
					line = br.readLine();
				}
			} catch (Exception e) {
				GlobalVariablesFunctions.showErrorMessage("Voxel location file " + file.getName() + " could not be loaded.");
				return;
			}
			
			mFileLoaded = true;
		}
	}
	
	public void saveFiles() {
		JFileChooser chooser = new JFileChooser(mDirectory);
		if (mNumLags == 1) {
			chooser.setFileFilter(new BfMRIVoxelDataFileFilter());
			mFileExtension = BfMRIVoxelDataFileFilter.EXTENSION;
		} else {
			chooser.setFileFilter(new fMRIVoxelDataFileFilter());
			mFileExtension = fMRIVoxelDataFileFilter.EXTENSION;
		}
		
		// Prompts the user to select a voxel data file to save as or to
		// type in the prefix of the file instead.
		int option = chooser.showDialog(mParent, "Enter the prefix of the Voxel Data File to save");
		if (option == JFileChooser.APPROVE_OPTION) {
			File file = chooser.getSelectedFile();
			mDirectory = file.getParent();
			mFilePrefix = file.getName();
			
			// We only want the prefix in case the user selected an existing file.
			int index;
			if ((index = mFilePrefix.indexOf(BfMRIVoxelDataFileFilter.EXTENSION)) != -1) {
				mFilePrefix = mFilePrefix.substring(0, index);
			} else if ((index = mFilePrefix.indexOf(fMRIVoxelDataFileFilter.EXTENSION)) != -1) {
				mFilePrefix = mFilePrefix.substring(0, index);
			}
			
		} else {
			return;
		}
		
		// Next, prompts the user for the neighborhood size.
		while (true) {
			String input = JOptionPane.showInputDialog(mParent,
				"If you would like to use the average intensity of the\n" +
				"neighborhood voxels as the intensity of voxels for each\n" +
				"inputted voxel location file, please enter neighborhood\n" +
				"size (the number of voxels from inputted voxel). Otherwise,\n" +
				"please enter 0.",
				"Voxel Neighborhood Size",
				JOptionPane.PLAIN_MESSAGE);
			if (input == null) {
				return;
			} else if (input.equals("")) {
				GlobalVariablesFunctions.showErrorMessage("No value was given.");
			} else {
				try {
					mNeighborhoodSize = (int) Math.round(Double.parseDouble(input));
					break;
				} catch (Exception e) {
					GlobalVariablesFunctions.showErrorMessage(input + " is not a valid value.");
				}
			}
		}
		
		// Extracts the intensity values of the datamats in a
		// separate process so the progress dialog can be
		// updated.
		final JDialog progressDialog = createProgressDialog();
		new Thread() {
			public void run() {
				extractFromDatamatFiles();
				progressDialog.dispose();
			}
		}.start();
	}
	
	private void extractFromDatamatFiles() {
		
		// Sets the values of the progress bar used to keep track
		// of the datamat extraction process here.
		mProgressBar.setValue(0);
		mProgressBar.setMinimum(0);
		mProgressBar.setMaximum(mDatamatProfiles.size() * 2 + 2);
		
		BufferedWriter bw;
		
		// An ArrayList representing all the values extracted from all the
		// datamat files. This is used to store the data to be outputted
		// as the one main voxel data file.
		ArrayList<ArrayList<String>> allValues = new ArrayList<ArrayList<String>>();
		
		Map<String, MLArray> currFile = null;
		
		for (int i = 0; i != mDatamatProfiles.size(); i++) {
			ArrayList<String> currDatamatProfiles = mDatamatProfiles.get(i);
			
			for (int j = 0; j != currDatamatProfiles.size(); j++) {
				String datamatName = currDatamatProfiles.get(j);
				
				// Reads in the datamat file.
				try {
					currFile = new NewMatFileReader(datamatName, new MatFileFilter()).getContent();
				} catch (Exception e) {
					GlobalVariablesFunctions.showErrorMessage("Datamat information could not be read from file " + datamatName + ".");
					continue;
				}
				
				// Obtains the variables we need from the datamat file.
				double[][] stDatamat = ((MLDouble) currFile.get("st_datamat")).getArray();
				int numConditions = stDatamat.length;
				int stWinSize = ((MLDouble)currFile.get("st_win_size")).getReal(0, 0).intValue();
				int[] stCoords = ((MLDouble) currFile.get("st_coords")).getIntFirstRowOfArray();
				int[] stDims = ((MLDouble) currFile.get("st_dims")).getIntFirstRowOfArray();
				
				// Determines the name of the voxel data file to be created for the
				// current datamat.
				String filename = mDirectory + File.separator + mFilePrefix + mFileExtension;
				filename = filename.substring(0, filename.indexOf("voxeldata.txt"));
				filename += "_grp" + (i + 1) + "_subj" + (j + 1) + "_voxeldata.txt";
				
				// Create ArrayLists where each one represents a single
				// row of each individual datamat file.
				while (allValues.size() < numConditions) {
					allValues.add(new ArrayList<String>());
				}
				
				try {
					
					// Writes the datamat intensity values to the voxel data file.
					bw = new BufferedWriter(new FileWriter(filename));
					
					String line;
					for (int row = 0; row != numConditions; row++) {
						line = "";
						
						// Goes through each given voxel location and retrieves the
						// intensity values from the current datamat.
						for (int k = 0; k != mVoxelLocations.size(); k++) {
							int[] voxelLocation = mVoxelLocations.get(k);
							int voxelLocation1D = (int)((stDims[0] * stDims[1] * (voxelLocation[2] - 1)) + (stDims[0] * (voxelLocation[1] - 1)) + (voxelLocation[0] - 1));
								
							// Determines the indices in the datamat that correspond
							// to the current voxel location.
							int index = 0;
							boolean found = false;
							for (index = 0; index != stCoords.length; index++) {
								if (stCoords[index] == voxelLocation1D + 1) {
									found = true;
									break;
								}
							}
							int temporalIndex = index * stWinSize;
							
							if (found) {
								for (int lag = 0; lag != stWinSize; lag++) {
									line += Double.toString(stDatamat[row][temporalIndex + lag]) + " ";
								}
							} else {
								for (int lag = 0; lag != stWinSize; lag++) {
									line += "0.0 ";
								}
							}
						}
						line = line.trim();
						allValues.get(row).add(line);
						bw.write(line);
						bw.newLine();
					}
					bw.close();
					
					mProgressBar.setValue(mProgressBar.getValue() + 2);
				} catch (Exception e) {
					GlobalVariablesFunctions.showErrorMessage("Voxel data file " + filename + " could not be created.");
					continue;
				}
			}
		}
		
		String filename = mDirectory + File.separator + mFilePrefix + mFileExtension;
		try {
			
			// Creates the main voxel data file containing the intensity
			// values from all the datamat files.
			bw = new BufferedWriter(new FileWriter(filename));
			for (int i = 0; i != allValues.size(); i++) {
				
				ArrayList<String> values = allValues.get(i);
				for (int j = 0; j != values.size(); j++) {
					bw.write(values.get(j));
					bw.newLine();
				}
			}
			bw.close();
		} catch (Exception e) {
			GlobalVariablesFunctions.showErrorMessage("Voxel data file " + filename + " could not be created.");
		}
		
		mProgressBar.setValue(mProgressBar.getMaximum());
		
		// The array of the total neighborhood voxel locations.
		/*ArrayList<ArrayList<ArrayList<int[]>>> neighborhoodVoxels = new ArrayList<ArrayList<ArrayList<int[]>>>();
		
		for (int i = 0; i != mDatamatProfiles.size(); i++) {
			ArrayList<String> currDatamatProfiles = mDatamatProfiles.get(i);
			int currSize = currDatamatProfiles.size();
			
			// The array of neighborhood voxel locations for the current
			// group's datamat profiles.
			ArrayList<ArrayList<int[]>> currGroupNeighborhoodVoxels = new ArrayList<ArrayList<int[]>>();
			
			for (int j = 0; j != currSize; j++) {
				
				// The array of neighborhood voxel locations for the current
				// datamat profile.
				ArrayList<int[]> currSubjectNeighborhoodVoxels = new ArrayList<int[]>();
				
				for (int k = 0; k != mVoxelLocations.size(); k++) {
					int[] voxelLocation = mVoxelLocations.get(k);
					
					// If the neighboorhood size given by the user is greater
					// than 0, retrieves all the neighbor voxel locations
					// relative to the neighborhood size.
					if (mNeighborhoodSize > 0) {
						int x1 = voxelLocation[1] - mNeighborhoodSize;
			            if (x1 < 1) { x1 = 1; }

			            int x2 = voxelLocation[1] + mNeighborhoodSize;
			            if (x2 > mDimensions[0]) { x2 = mDimensions[0]; }

			            int y1 = voxelLocation[2] - mNeighborhoodSize;
			            if (y1 < 1) { y1 = 1; }

			            int y2 = voxelLocation[2] + mNeighborhoodSize;
			            if (y2 > mDimensions[1]) { y2 = mDimensions[1]; }

			            int z1 = voxelLocation[3] - mNeighborhoodSize;
			            if (z1 < 1) { z1 = 1; }

			            int z2 = voxelLocation[3] + mNeighborhoodSize;
			            if (z2 > mDimensions[2]) { z2 = mDimensions[2]; }
			            
			            for (int o = z1; o != z2 + 1; o++) {
			            	for (int n = y1; n != y2 + 1; n++) {
			            		for (int m = x1; m != x2 + 1; m++) {
			            			int[] neighborVoxelLocation = new int[] {voxelLocation[0], m, n, o};
			            			currSubjectNeighborhoodVoxels.add(neighborVoxelLocation);
			            		}
			            	}
			            }
			        
			        // Otherwise, adds the one voxel location only.
					} else {
						currSubjectNeighborhoodVoxels.add(voxelLocation);
					}
				}
				currGroupNeighborhoodVoxels.add(currSubjectNeighborhoodVoxels);
			}
			neighborhoodVoxels.add(currGroupNeighborhoodVoxels);
		}
		
		Map<String, MLArray> currFile = null;
		for (int i = 0; i != mDatamatProfiles.size(); i++) {
			ArrayList<String> currDatamatProfiles = mDatamatProfiles.get(i);
			int currSize = currDatamatProfiles.size();
			for (int j = 0; j != currSize; j++) {
				String datamatName = currDatamatProfiles.get(j);
				
				// Reads in the datamat file.
				try {
					currFile = new NewMatFileReader(datamatName, new MatFileFilter()).getContent();
				} catch (Exception e) {
					showErrorDialog("Datamat information could not be read from file " + datamatName + ".");
					continue;
				}
				
				// Obtains the variables we need from the datamat file.
				int[] stCoords = ((MLDouble) currFile.get("st_coords")).getIntFirstRowOfArray();
				double[][] stDatamat = ((MLDouble) currFile.get("st_datamat")).getArray();
				
				double[] temp = MLFuncs.rangeDouble(1, mNumLags * mCoordinates.length);
				double[][] temp2 = MLFuncs.reshape(temp, mNumLags, mCoordinates.length);
				
				
			}
		}*/
	}
	
	// Creates the progress dialog used to display the progress of the voxels being
	// extracted from each datamat file.
	private JDialog createProgressDialog() {
		JDialog dialog = new JDialog(mParent, "Extracting Voxels");
		SpringLayout s = new SpringLayout();
		dialog.setLayout(s);
		
		JLabel progressBarMessage = new JLabel("Please wait...");
		
		dialog.add(progressBarMessage);
		dialog.add(mProgressBar);
		
		s.putConstraint(SpringLayout.NORTH, progressBarMessage, 5, SpringLayout.NORTH, dialog);
		s.putConstraint(SpringLayout.WEST, progressBarMessage, 5, SpringLayout.WEST, dialog);
		s.putConstraint(SpringLayout.NORTH, mProgressBar, 5, SpringLayout.SOUTH, progressBarMessage);
		s.putConstraint(SpringLayout.WEST, mProgressBar, 5, SpringLayout.WEST, dialog);
		
		int width = 200;
		int height = 80;
		dialog.setSize(width, height);
		
		int x = (Toolkit.getDefaultToolkit().getScreenSize().width / 2) - (width / 2);
		int y = (Toolkit.getDefaultToolkit().getScreenSize().height / 2) - (height / 2);
		dialog.setLocation(x, y);

		dialog.setAlwaysOnTop(true);
		dialog.setVisible(true);
		
		return dialog;
	}
	
	// Parses each given line, which is taken from the voxel location
	// file, and adds the parsed values to an array to store them.
	// Returns false if the line can not be parsed, or if it contains
	// invalid values.
	private boolean areVoxelCoordsValid(String filename, String line) {
		String[] coords = line.split(" ");
		int[] voxelLocation = new int[3];
		
		// There should only be the three coordinates given.
		if (coords.length == 3) {
			
			for (int i = 0; i != coords.length; i++) {
				try {
					
					// Each value is given as a coordinate in millimeters
					// so it needs to be parsed and then converted to a voxel.
					double coord = Double.parseDouble(coords[i]);
					coord = coord / mVoxelSize[i] + mOrigin[i];
					int voxel = (int) Math.round(coord);
					if (voxel < 1 || voxel > mDimensions[i]) {
						GlobalVariablesFunctions.showErrorMessage("Voxel location file " + filename
								+ " contains an out-of-bounds coordinate: " + coords[i]);
						return false;
					}
					
					voxelLocation[i] = voxel;
				} catch (Exception e) {
					GlobalVariablesFunctions.showErrorMessage("Voxel location file " + filename
							+ " contains an invalid value: " + coords[i]);
					return false;
				}
			}
			
		} else if (coords.length < 3) {
			GlobalVariablesFunctions.showErrorMessage("Voxel location file " + filename
					+ " does not contain enough values on each line.");
			return false;
			
		} else {
			GlobalVariablesFunctions.showErrorMessage("Voxel location file " + filename
					+ " contains too many values on each line.");
			return false;
			
		}
		
		// If this point is reached, then the given line was parsed correctly.
		// Checks that the voxel location is within the brain.
		int voxelLocation1D = mCurrentBrainData.convert3Dto1D(voxelLocation[0], voxelLocation[1], voxelLocation[2]);
		if (!mCoordinates.contains(voxelLocation1D)) {
			GlobalVariablesFunctions.showErrorMessage("The voxel location," + line + ", is not part of the brain.");
			return false;
		}
		
		mVoxelLocations.add(voxelLocation);
		return true;
	}
	
	public boolean isVoxelLocationFileLoaded() {
		return mFileLoaded;
	}
	
}
