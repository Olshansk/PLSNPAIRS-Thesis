package pls.chrome.result.blvplot;

import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Iterator;
import java.util.TreeSet;

import javax.swing.BoxLayout;
import javax.swing.DefaultListModel;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JList;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;

import npairs.Npairsj;
import npairs.io.NiftiIO;

import pls.chrome.result.controller.ResultsCommandManager;
import pls.chrome.result.model.BrainFilter;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.ResultModel;
import pls.shared.AnalyzeImageFileFilter;
import pls.shared.GlobalVariablesFunctions;
import pls.shared.MLFuncs;
import pls.shared.NiftiImageFileFilter;

@SuppressWarnings("serial")
public class FilterBrowser extends JPanel implements MouseListener, ActionListener{
	private GeneralRepository mRepository;
	
	private JList mList;
	private DefaultListModel mListModel = new DefaultListModel();
	private JButton removeButton;
	private JButton saveButton;

	public FilterBrowser(GeneralRepository repository) {
		mRepository = repository;
		
		setLayout(new BoxLayout(this, BoxLayout.Y_AXIS) );
		
		mList = new JList(mListModel);
		
		mList.addMouseListener(this);
		
		JScrollPane scroller = new JScrollPane(mList);
		scroller.setAlignmentX(JScrollPane.LEFT_ALIGNMENT);
		scroller.setPreferredSize(new Dimension(300, scroller.getPreferredSize().height));

		add(scroller);
		
		FilterBrowserCellRenderer renderer = new FilterBrowserCellRenderer();
		mList.setCellRenderer(renderer);
		
		JPanel panel = new JPanel();
		panel.setLayout(new BoxLayout(panel, BoxLayout.X_AXIS) );
		panel.setAlignmentX(JPanel.LEFT_ALIGNMENT);
		
		JButton loadButton = new JButton("Add mask");
		loadButton.setActionCommand("load filter");
		loadButton.addActionListener(this);
		panel.add(loadButton);
		
		removeButton = new JButton("Remove mask");
		removeButton.setEnabled(false);
		removeButton.setActionCommand("remove filter");
		removeButton.addActionListener(this);
		panel.add(removeButton);
		
		saveButton = new JButton("Save mask");
		saveButton.setEnabled(false);
		saveButton.setActionCommand("save filter");
		saveButton.addActionListener(this);
		panel.add(saveButton);
		
		add(panel);
	}

	/**
	 * Called whenever a brain filter event occurs. Called from
	 * LeftSidePanel.java
	 */
	public void refresh() {
		BrainFilter brainFilter = mRepository.getBrainFilter();
		
		mListModel.clear();
		
		for (String filterName : brainFilter.filterNames() ) {
			FilterBrowserItem item = new FilterBrowserItem(filterName);
			item.beingDisplayed = brainFilter.isEnabled(filterName);
			mListModel.addElement(item);
		}

		//Disable save/remove buttons if there are no files loaded.
		if(mListModel.isEmpty()){
			saveButton.setEnabled(false);
			removeButton.setEnabled(false);
		}else{
			//items are available but none is selected to save.
			if(mList.getSelectedIndex() == -1){
				saveButton.setEnabled(false);
			}
			removeButton.setEnabled(true);
		}
	}

	@Override
	public void mouseClicked(MouseEvent e) {}

	@Override
	public void mouseEntered(MouseEvent e) {}

	@Override
	public void mouseExited(MouseEvent e) {}

	@Override
	public void mousePressed(MouseEvent evt) {
		JList list = (JList)evt.getSource();
        if (evt.getClickCount() == 2) {          // Double-click
            // Get item index
            int index = list.locationToIndex(evt.getPoint());
            
            String filterName = mListModel.get(index).toString();
            
            ResultsCommandManager.toggleFilterEnabled(filterName);
        }
		if(mList.getSelectedIndex() == -1){
			saveButton.setEnabled(false);
		}else{
			saveButton.setEnabled(true);
		}

	}

	@Override
	public void mouseReleased(MouseEvent e) {}

	@Override
	public void actionPerformed(ActionEvent e) {
		if (e.getActionCommand().equals("load filter") ) {
			loadFilterAction();
		} else if (e.getActionCommand().equals("remove filter") ) {
			//Did the user select a value?
			if(mList.getSelectedValue() == null) return;
			
			String filter = mList.getSelectedValue().toString();
			mRepository.removeBrainFilter(filter);

		} else if (e.getActionCommand().equals("save filter")) {
			saveFilterAction();
		}
	}
	
	/**
	 * Action performed when the user hits the "Add Mask" button.
	 */
	private void loadFilterAction(){
		JFileChooser chooser = new JFileChooser();
		chooser.addChoosableFileFilter(new AnalyzeImageFileFilter());
		chooser.addChoosableFileFilter(new NiftiImageFileFilter());

		int option = chooser.showDialog(this, "Load Mask File");

		if (option == JFileChooser.APPROVE_OPTION) {
			String imageDataPath = chooser.getSelectedFile().getParent();
			String imageFileName = chooser.getSelectedFile().getName();


			double[] imageData = null;
			try {
				imageData = getImageData(imageDataPath, imageFileName);
			} catch (Exception ex) {
				GlobalVariablesFunctions.showErrorMessage("Brain mask image " +
						"file" + " " + imageFileName + " could not be loaded.");

				return;
			}

			if (imageData != null) {
				/*String thresholdString = JOptionPane.showInputDialog(
						"Brain mask image file has been scaled so that " +
						"it goes from 0 to 1.\nPlease enter a " +
						"threshold for ON values.", 0);
				double threshold = 0;
				try {
					threshold = Double.parseDouble(thresholdString);
				} catch (NumberFormatException nfex) {
					GlobalVariablesFunctions.showErrorMessage("Invalid " +
							"threshold.");
					return;
				}*/

				/*Check if we are loading a mask that is already loaded but
				 * under a different extension. i.e we are loading mask.hdr
				 * when mask.img has already been loaded*/
				if(checkLoadedPermutation(imageFileName)){
					int index = imageFileName.indexOf(".img");
					String remove;
					
					if(index == -1){
						index = imageFileName.indexOf(".hdr");
						remove = imageFileName.substring(0, index);
						remove = remove + ".img";
						
					}else{
						remove = imageFileName.substring(0, index);
						remove = remove + ".hdr";
					}
					mRepository.removeBrainFilter(remove);
				}
				
				TreeSet<Integer> coords = new TreeSet<Integer>();

				for (int i = 0; i < imageData.length; ++i) {
					if (imageData[i] == 1.0) {
						coords.add(i);
					}
				}

				mRepository.addBrainFilter(imageFileName, coords);
			}
		}
	}

	/**
	 * Determine if the mask the user is attempting to load has already been
	 * loaded under a different extension. I.e if mask.img has been loaded
	 * and the user attemps to load mask.hdr this is the same mask so don't
	 * consider it a seperate mask and load it simultaneously.
	 * @param imageFileName the filename of the mask being loaded.
	 * @return true if the mask has already been loaded. false otherwise.
	 */
	private boolean checkLoadedPermutation(String imageFileName){


		BrainFilter brainFilter = mRepository.getBrainFilter();
		int index = imageFileName.indexOf(".img");
		String bareName;

		if(index != -1){
			bareName = imageFileName.substring(0, index);
			if(brainFilter.filterNames().contains(bareName + ".hdr")){
				return true;
			}
			return false;
		}

		index = imageFileName.indexOf(".hdr");
		if(index != -1){
			bareName = imageFileName.substring(0, index);
			if(brainFilter.filterNames().contains(bareName + ".img")){
				return true;
			}
			return false;
		}

		//filename is of some other form so allow it to be loaded.
		return false;
	}
	/**
	 * Action performed when the user hits the "Save Mask" button.
	 */
	private void saveFilterAction(){
		JFileChooser chooser = new JFileChooser();
		AnalyzeImageFileFilter analyzeFilter = new AnalyzeImageFileFilter();
		NiftiImageFileFilter niftiFilter = new NiftiImageFileFilter();
		chooser.addChoosableFileFilter(analyzeFilter);
		chooser.addChoosableFileFilter(niftiFilter);

		int option = chooser.showSaveDialog(this);

		if (option == JFileChooser.APPROVE_OPTION) {
			String filename = chooser.getSelectedFile().getAbsolutePath();
			TreeSet<Integer> mask = getSelectedFilter();

			if(chooser.getFileFilter().equals(niftiFilter))
				filename = filename + ".nii";
			
			int[] dims = mRepository.getGeneral().getDimensions();
			double[] image = new double[dims[0] * dims[1] * dims[3]];

			Iterator<Integer> iter = mask.iterator();
			if (iter.hasNext()) {
				int currNum = iter.next();
				for (int i = 0; i < image.length; ++i) {
					if (i == currNum) {
						image[i] = 1;
						if (iter.hasNext()) {
							currNum = iter.next();
						}
					} else {
						image[i] = 0;
					}
				}
			}

			double[][][] image3D = reshape(image, dims[3], dims[1], dims[0]);

			try {
				if (!checkOverwrite(filename)) return;
				// data saved in INT16 format
				// TODO: Could be in binary or some other format?
				NiftiIO.writeVol(image3D, filename);
			} catch (FileNotFoundException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			} catch (IOException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
		}
	}

	/**
	 * Determine whether or not saving the currently selected mask means
	 * overwriting another already existing mask.
	 * @param filename the filename to save the selected mask to.
	 * @return true if the user wants to overwrite or there is nothing to
	 * overwrite to. false otherwise.
	 */
	private boolean checkOverwrite(String filename){
		//both analyze and nifti files are written out as hdr/img pairs
		//this is the current way things are done and thus this is the
		//rationalization for not checking .nii
		File f = new File(filename + ".img");
		File f2 = new File(filename + ".hdr");
		String message = null;
		if (f.exists() && f2.exists()) {
			message = "Do you wish to overwrite " + f.getName() + "/"
					+ f2.getName() + " ?";
		} else if (f.exists()) {
			message = "Do you wish to overwrite " + f.getName() + " ?";
		} else if (f2.exists()) {
			message = "Do you wish to overwrite " + f2.getName() + " ?";
		}

		if (message != null) {
			int selection = JOptionPane.showConfirmDialog(this, message);

			if (selection != JOptionPane.OK_OPTION) {
				return false;
			}
		}
		return true;
	}
	
	/**
	 * Precondition: A filter has been selected to be saved. This should be
	 * the case because this function is called when the save button is hit
	 * and the save button is only enabled when a filter has been selected.
	 * 2. The filter exists in the brain filter collection (brainFilter).
	 * @return The set of coordinates that belong to the selected filter.
	 */
	private TreeSet<Integer> getSelectedFilter(){
		BrainFilter brainFilter = mRepository.getBrainFilter();
		FilterBrowserItem filter = (FilterBrowserItem) mList.getSelectedValue();
		String filterName = filter.toString();

		return brainFilter.getFilter(filterName);
	}

	private double[][][] reshape(double[] data, int x, int y, int z) {
		double[][][] ret = new double[x][y][z];
		int col = 0;
		for (int i = 0; i != x; i++) {
			for (int j = 0; j != y; j++) {
				for (int k = 0; k != z; k++) {
					ret[i][j][k] = data[col];
					col++;
				}
			}
		}
		return ret;
	}
		
	private double[] getImageData(String folder, String file) {
		ResultModel model = mRepository.getGeneral();
		int[] stDims = model.getBrainData().getDimensions();
		
		int[] bgDims = null;
		double[] bgImageData = null;
		try {
			bgDims = NiftiIO.getVolDims3D(folder + "/" + file);
			bgImageData = NiftiIO.readNiftiData(folder + "/" + file, 0);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		

		// Compares the resolutions of the background image and brain image.
		// Note: For the background image, its resolution is stored as [x y z t]
		// while the brain image's resolution is stored as [x y t z].
//		int[] bgDims = bgImage.get4DDimensions();
		if (bgDims[0] != stDims[0] || bgDims[1] != stDims[1]
				|| bgDims[2] != stDims[3]) {
			System.out.println(bgDims[0] + ", " + bgDims[1] + ", " + bgDims[2]);
			System.out.println("vs");
			System.out.println(stDims[0] + ", " + stDims[1] + ", " + stDims[3]);
			GlobalVariablesFunctions.showErrorMessage("Mask image file "
													+ "does not have a matching " +
													"resolution to the brain image.");
			return null;
		}

//		double[] bgImageData = bgImage.getData();
		bgImageData = MLFuncs.normalize(bgImageData);
		
		return bgImageData;
	}
}
