package pls.chrome.result.blvplot;

import java.awt.Color;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.FocusEvent;
import java.awt.event.FocusListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.util.ArrayList;

import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.border.LineBorder;

import pls.chrome.result.controller.ResultsCommandManager;
import pls.chrome.result.model.BrainData;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.ResultModel;
import pls.shared.GlobalVariablesFunctions;

@SuppressWarnings("serial")
public class CoordinatesPanel extends JPanel implements ActionListener, KeyListener, FocusListener {
	private JComboBox mLagComboBox = null;
	private JButton mGoButton = null;
	private JLabel mPositionLabels[] = null;
	private JTextField[] mPositionVoxelTextFields = null;	
	private JTextField[] mPositionMmTextFields = null;
	
	private ArrayList<ArrayList<Double> > mValidMmPositions = null;
	private GeneralRepository mRepository = null;
	private String lastSelectedModel = "";

	public CoordinatesPanel(GeneralRepository repository) {
		setAlignmentX(JPanel.LEFT_ALIGNMENT);
		mRepository = repository;
		ResultModel model = repository.getGeneral();
		lastSelectedModel = mRepository.getSelectedResultFile();

		mValidMmPositions = makeValidMmPositions(model.getBrainData() );
		
		setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
		setBorder(new LineBorder(Color.DARK_GRAY, 1) );
		
		JLabel title = new JLabel("Selected Voxel");
		Font font = title.getFont();
		// Make the title have a bold and slightly larger font
		title.setFont(new Font(font.getFontName(), Font.BOLD, font.getSize() + 1));
		title.setAlignmentX(JLabel.LEFT_ALIGNMENT);
		
//		add(title);
		
		JPanel gridPanel = new JPanel();
		gridPanel.setLayout(new GridLayout(4, 4));
		gridPanel.setAlignmentX(JPanel.LEFT_ALIGNMENT);
		
		// Create widgets to view and set the current position
		JLabel lagLabel = new JLabel("Lag #: ");
		lagLabel.setHorizontalAlignment(JLabel.RIGHT);
		JLabel voxelLabel = new JLabel("In voxels: ");
		voxelLabel.setHorizontalAlignment(JLabel.RIGHT);
		JLabel mmLabel = new JLabel("In mm: ");
		mmLabel.setHorizontalAlignment(JLabel.RIGHT);
		
		mGoButton = new JButton("Go");
		mLagComboBox = new JComboBox();
		
		mPositionLabels = new JLabel[3];
		for (int i = 0; i < 3; ++i) {
			mPositionLabels[i] = new JLabel();
			mPositionLabels[i].setHorizontalAlignment(JLabel.CENTER);
		}
		mPositionLabels[0].setText("X");
		mPositionLabels[1].setText("Y");
		mPositionLabels[2].setText("Z");
		
		mPositionVoxelTextFields = new JTextField[3];
		for (int i = 0; i < 3; ++i) {
			mPositionVoxelTextFields[i] = new JTextField(4);
		}
		mPositionMmTextFields = new JTextField[3];
		for (int i = 0; i < 3; ++i) {
			mPositionMmTextFields[i] = new JTextField(4);
		}
		
		// We want the grid to look like this:
		// [blank][X]   [Y]    [Z]
		// [voxel][xvox][yvox] [zvox]
		// [in mm][xmm] [ymm]  [zmm]
		// [lag #][lag] [blank][GO]
		
		gridPanel.add(new JLabel() );
		for (JLabel label : mPositionLabels) {
			gridPanel.add(label);
		}
		gridPanel.add(voxelLabel);
		for (JTextField field : mPositionVoxelTextFields) {
			gridPanel.add(field);
		}
		gridPanel.add(mmLabel);
		for (JTextField field : mPositionMmTextFields) {
			gridPanel.add(field);
		}
		gridPanel.add(lagLabel);
		gridPanel.add(mLagComboBox);
		gridPanel.add(new JLabel() );
		gridPanel.add(mGoButton);
		
		// Do not display the combo box for selecting a lag if there is
		// only one lag available.
//		if (numLags <= 1) {
//			lagLabel.setVisible(false);
//			mLagComboBox.setVisible(false);
//		}
		
		add(gridPanel);
		
		updateSelection();
		
		mGoButton.addActionListener(this);
		
		
		// Add focus listeners and key listeners for some of the text fields
		for (int i = 0; i < 3; ++i) {
			mPositionVoxelTextFields[i].addFocusListener(this);
			mPositionVoxelTextFields[i].addKeyListener(this);
			mPositionMmTextFields[i].addFocusListener(this);
			mPositionMmTextFields[i].addKeyListener(this);
		}
		
		updateLagComboBox();
		
//		int numLags = model.getWindowSize();
//		for (int i = 0; i < numLags ; ++i)
//		{
//			mLagComboBox.addItem(Integer.toString(i) );
//		}
		
		// Creates three sets of lag numbers, with each set representing a
		// single brain view's visible lags. The default is having each set
		// contain all the lag numbers, which is from 0 to the number of
		// lags - 1, since the lag values are 0-based still.
//		for (int i = 0; i < 3; ++i) {
//			ArrayList<Integer> lags = new ArrayList<Integer>();
//			for (int j = 0; j < numLags; ++j) {
//				lags.add(j);
//			}
//			mLagNumbers.add(lags);
//		}
	}
	
	public void updateBrainView() {
		updateLagComboBox();
		updateSelection();
	}
	
	/**
	 * Updates the selectable lags in the lag combo box so only those lags
	 * that are visible are selectable.
	 */
	public void updateLagComboBox() {
		int brainView = mRepository.getImagePropertiesModel().getBrainView();
		
		// Updates the combo box for selecting lags such that it only
		// contains the lags that are visible for the current file.
		String currentModel = mRepository.getSelectedResultFile();

		ArrayList<Integer> lagNumbers = mRepository.getControlPanelModel().
				getViewableLags(currentModel, brainView);
		
		mLagComboBox.removeActionListener(this);
		mLagComboBox.removeAllItems();
		for (int i = 0; i < lagNumbers.size(); ++i)
		{
			mLagComboBox.addItem(Integer.toString(lagNumbers.get(i)));
		}
		mLagComboBox.addActionListener(this);
	}

	public void actionPerformed(ActionEvent event) {
		selectVoxel();
	}

	public void keyPressed(KeyEvent e) {
		// We only handle the case where the enter key has been pressed.
		if (e.getKeyChar() != BrainLatentVariablesPlot.ENTER_KEY) {
			return;
		}
		
		if (e.getSource() == mPositionVoxelTextFields[0]
		    || e.getSource() == mPositionVoxelTextFields[1]
		    || e.getSource() == mPositionVoxelTextFields[2]) {
			selectVoxel();
		}
		else if (e.getSource() == mPositionMmTextFields[0]
		    || e.getSource() == mPositionMmTextFields[1]
		    || e.getSource() == mPositionMmTextFields[2]) {
			synchronizePositionFields(true);
			selectVoxel();
		}		
	}

	public void keyReleased(KeyEvent e) {
	}

	public void keyTyped(KeyEvent e) {
	}

	public void focusGained(FocusEvent arg0) {
	}

	public void focusLost(FocusEvent event) {
		// When the user has finished editing a position in voxels text
		// field, adjust the position in mm text fields.
		if (event.getSource() == mPositionVoxelTextFields[0] ||
				event.getSource() == mPositionVoxelTextFields[1] ||
				event.getSource() == mPositionVoxelTextFields[2]) {
			synchronizePositionFields(false);
		}
		// When the user has finished editing a position in mm text
		// field, adjust the position in voxels text fields.
		else if (event.getSource() == mPositionMmTextFields[0] ||
				event.getSource() == mPositionMmTextFields[1] ||
				event.getSource() == mPositionMmTextFields[2]) {
			synchronizePositionFields(true);
		}
	}
	
	private void synchronizePositionFields(boolean mmAsSource) {
		if (mmAsSource) {
			for (int i = 0; i < 3; ++i) {
				double val;
				try {
					val = Double.parseDouble(mPositionMmTextFields[i].getText() );
				}
				catch (NumberFormatException e) {
					return;
				}
				
				int adjustVal = adjustMmPosition(val, mValidMmPositions.get(i) );
				
				// Add 1 because matlab (and our volume) is 1 based.
				mPositionVoxelTextFields[i].setText(Integer.toString(adjustVal + 1) );
				mPositionMmTextFields[i].setText(Double.toString(mValidMmPositions.get(i).get(adjustVal) ) );
			}
		}
		else {
			for (int i = 0; i < 3; ++i) {
				int val;
				
				try {
					val = Integer.parseInt(mPositionVoxelTextFields[i].getText() );
				}
				catch (NumberFormatException e) {
					return;
				}
				
				val = Math.max(val, 1);
				val = Math.min(val, mValidMmPositions.get(i).size() );
				
				mPositionVoxelTextFields[i].setText(Integer.toString(val) );
				mPositionMmTextFields[i].setText(Double.toString(mValidMmPositions.get(i).get(val - 1) ) );
			}
		}
	}
	
	/**
	 * Find the closest valid mm position to what the user has typed in, and
	 * return its corresponding voxel number.
	 */
	private int adjustMmPosition(double val, ArrayList<Double> array) {
		int voxelIndex = 0;
		
		// Search the array in descending order
		for (int i = array.size() - 1; i > 0; --i) {
			if (array.get(i) <= val) {
				voxelIndex = i;
				break;
			}
		}
		
		if (voxelIndex < array.size() - 1) {
			double diff1 = Math.abs(array.get(voxelIndex) - val);
			double diff2 = Math.abs(array.get(voxelIndex + 1) - val);
			
			if (diff2 <= diff1) {
				++voxelIndex;
			}
		}
		
		return voxelIndex;
	}
	
	public void selectVoxel() {
		int x, y, z;
		
		int lag = Integer.parseInt((String) mLagComboBox.getSelectedItem());
		
		try {
			int[] dimensions = mRepository.getGeneral().getBrainData().getDimensions();
			
			String xMmValue = mPositionMmTextFields[0].getText().trim();
			String yMmValue = mPositionMmTextFields[1].getText().trim();
			String zMmValue = mPositionMmTextFields[2].getText().trim();
			
			if (xMmValue.equals("")) {
				GlobalVariablesFunctions.showErrorMessage("The mm value for x is empty.");
				return;
			} else if (yMmValue.equals("")) {
				GlobalVariablesFunctions.showErrorMessage("The mm value for y is empty.");
				return;
			} else if (zMmValue.equals("")) {
				GlobalVariablesFunctions.showErrorMessage("The mm value for z is empty.");
				return;
			}
			
			String xVoxelValue = mPositionVoxelTextFields[0].getText().trim();
			String yVoxelValue = mPositionVoxelTextFields[1].getText().trim();
			String zVoxelValue = mPositionVoxelTextFields[2].getText().trim();
			
			if (xVoxelValue.equals("")) {
				GlobalVariablesFunctions.showErrorMessage("The voxel value for x is empty.");
				return;
			} else if (yVoxelValue.equals("")) {
				GlobalVariablesFunctions.showErrorMessage("The voxel value for y is empty.");
				return;
			} else if (zVoxelValue.equals("")) {
				GlobalVariablesFunctions.showErrorMessage("The voxel value for z is empty.");
				return;
			}
			
			x = Integer.parseInt(xVoxelValue);
			y = Integer.parseInt(yVoxelValue);
			z = Integer.parseInt(zVoxelValue);
			
			if (x < 1 || x > dimensions[0] ||
					y < 1 || y > dimensions[1] ||
					z < 1 || z > dimensions[3]) {
				GlobalVariablesFunctions.
						showErrorMessage("Voxel indices out of bounds. The dimensions of this volume are "
						+ dimensions[0] + " x " + dimensions[1] + " x " + dimensions[3]);
				return;
			}
		}
		catch (NumberFormatException e) {
			GlobalVariablesFunctions.showErrorMessage("Voxel indices must be integers.");
			return;
		}
		
		ResultsCommandManager.selectVoxel(lag, x, y, z);
	}
	
	/**
	 * Called by the selection commands.  Updates the text fields with the
	 * values from the command.
	 */
	public void updateSelection() {
		//may be able to speed this up if we choose to remember the last selected
		//file. if the selected file is now different then we update the combobox.
		//also give updated spec for voxel selection.

		ResultModel model = mRepository.getGeneral();
		String currentModel = mRepository.getSelectedResultFile();
		int[] dimensions = model.getBrainData().getDimensions();
		int[] selectedVoxel = model.getSelectionModel().getSelectedVoxel();
		
		int x = selectedVoxel[0];
		int y = selectedVoxel[1];
		int z = selectedVoxel[2];
		int lag = selectedVoxel[3];

		//This is here for efficiency, only repopulate the lag combobox
		//if we know we have switched selected models and hence the lags
		//selectable may be different.
		if (!lastSelectedModel.equals(currentModel)){
			updateLagComboBox();
			lastSelectedModel = currentModel;
		}
		
		// If the pixels are within the valid dimensions, update values
		if (x > 0 && x <= dimensions[0] &&
				y > 0 && y <= dimensions[1] &&
				z > 0 && z <= dimensions[3] ) {
			
			mPositionVoxelTextFields[0].setText(Integer.toString(x) );
			mPositionVoxelTextFields[1].setText(Integer.toString(y) );
			mPositionVoxelTextFields[2].setText(Integer.toString(z) );
			
			// Need to subtract 1 to access our zero based arrays
			mPositionMmTextFields[0].setText(Double.toString(mValidMmPositions.get(0).get(x - 1) ) );
			mPositionMmTextFields[1].setText(Double.toString(mValidMmPositions.get(1).get(y - 1) ) );
			mPositionMmTextFields[2].setText(Double.toString(mValidMmPositions.get(2).get(z - 1) ) );

			mLagComboBox.removeActionListener(this);
			mLagComboBox.setSelectedItem(Integer.toString(lag));
			mLagComboBox.addActionListener(this);
		}
		// Otherwise clear the values
		else {
			mPositionVoxelTextFields[0].setText("");
			mPositionVoxelTextFields[1].setText("");
			mPositionVoxelTextFields[2].setText("");
			
			mPositionMmTextFields[0].setText("");
			mPositionMmTextFields[1].setText("");
			mPositionMmTextFields[2].setText("");
		}
	}
	
	/**
	 * Fills arrays of the valid positions in mm.  Useful for when we need
	 * to correct input from the user.
	 */
	private ArrayList<ArrayList<Double> > makeValidMmPositions(BrainData bData) {
		int origin[] = bData.getOrigin();
		double size[] = bData.getVoxelSize();
		
		ArrayList<ArrayList<Double> > validMmPositions = new ArrayList<ArrayList<Double> >();
		
		ArrayList<Double> currList = new ArrayList<Double>();
		// x
		for (int i = 0; i < bData.getWidth(BrainData.AXIAL); ++i) {
			currList.add( ( (i + 1) - origin[0]) * size[0]);
		}
		validMmPositions.add(currList);
		
		currList = new ArrayList<Double>();
		// y
		for (int i = 0; i < bData.getHeight(BrainData.AXIAL); ++i) {
			currList.add( ( (i + 1) - origin[1]) * size[1]);
		}
		validMmPositions.add(currList);
		
		currList = new ArrayList<Double>();
		// z
		for (int i = 0; i < bData.getNumSlices(BrainData.AXIAL); ++i) {
			currList.add( ( (i + 1) - origin[2]) * size[2]);
		}
		validMmPositions.add(currList);
		
		return validMmPositions;
	}
	
//	public void setLagNumbers(int brainView, ArrayList<Integer> lagNumbers) {
//		mLagNumbers.set(brainView, lagNumbers);
//		updateLagComboBox(brainView);
//		updateSelection();
//	}

	public void volumesAdded() {
		updateLagComboBox();
		mValidMmPositions = makeValidMmPositions(mRepository.getGeneral().getBrainData() );
		updateSelection();
	}
}
