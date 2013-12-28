package pls.chrome.result.blvplot;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.util.ArrayList;

import java.util.HashMap;
import javax.swing.BoxLayout;
import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JSeparator;
import javax.swing.JTextField;
import javax.swing.border.LineBorder;

import pls.chrome.result.controller.ResultsCommandManager;
import pls.chrome.result.controller.observer.Event;
import pls.chrome.result.controller.observer.filters.BrainFilterEvent;
import pls.chrome.result.controller.observer.filters.FiltersObserver;
import pls.chrome.result.controller.observer.filters.IncorrectLagsSelectedEvent;
import pls.chrome.result.controller.observer.filters.SliceFiltersEvent;
import pls.chrome.result.controller.observer.filters.ViewedLvsEvent;
import pls.chrome.result.model.BrainData;
import pls.chrome.result.model.GeneralRepository;
import pls.shared.GlobalVariablesFunctions;


@SuppressWarnings("serial")
class ControlPanel extends JPanel implements ActionListener, KeyListener,
FiltersObserver {
	
	public static final int ENTER_KEY = 10;
	
	public static final String ZERO = "0";
	public static final String ONE = "1";
	
	public static final String LAG = "lag";
	public static final String SLICE = "slice";
	
	// The regular expressions used to for parsing the input fields related
	// to lag and slice selection.
	public static final String NUMBER_REGEX = "\\d+";
	public static final String HYPHEN_RANGE_REGEX = "\\d+\\s*-\\s*\\d+";
	public static final String COMMA_RANGE_REGEX = "(\\d+\\s*,\\s*)+\\d+";
	public static final String MULTI_RANGE_REGEX = "(\\d+|\\d+\\s*-\\s*\\d+)(\\s*,\\s*(\\d+|\\d+\\s*-\\s*\\d+))+";
	
	// String arrays used to store the values given in the input text
	// fields that are related to brain slice selection.
//	public String[] firstSlice = new String[3];
//	public String[] step = new String[3];
//	public String[] lastSlice = new String[3];
//	public String[] sliceRange = new String[3];
//	public String[] numRowsPerLag = new String[3];
//	public String[] lags = new String[3];
	
	// boolean array that indicates which button related to slice instruction
	// is selected in each brain view.
	public boolean[] sliceInstructionButton1Selected = {true, true, true};

//	public JComboBox brainViewBox = new JComboBox();
	
	public CrosshairPropertiesDialog crosshairDialog;// = new CrosshairPropertiesDialog();
	public JButton crosshairButton = new JButton("Crosshair Properties");
	
	public LabelPropertiesDialog labelDialog;// = new LabelPropertiesDialog();
	public JButton labelButton = new JButton("Label Properties");
	
	public JRadioButton sliceInstructionButton1 = new JRadioButton("Select slices by step value", true);
	public JRadioButton sliceInstructionButton2 = new JRadioButton("Select slices by manual input", false);
	
	public JLabel firstSliceLabel = new JLabel("First Slice: ");
	public JLabel stepLabel = new JLabel("Step: ");
	public JLabel lastSliceLabel = new JLabel("Last Slice: ");
	public JLabel sliceInputExampleLabel = new JLabel("For example, type 1, 3, 5-9");
	public JLabel displayAllLagsLabel = new JLabel("Display all lags:");

	public JTextField numRowsField = new JTextField(5);
	public JTextField firstSliceField = new JTextField(5);
	public JTextField stepField = new JTextField(5);
	public JTextField lastSliceField = new JTextField(5);
	public JTextField sliceInputField = new JTextField(15);
	public JTextField lagsField = new JTextField(7);

	public JCheckBox allLagsCheckBox = new JCheckBox();

	public JButton plotButton = new JButton("PLOT");
	public JButton resetButton = new JButton("RESET");

	GeneralRepository mRepository = null;
	
//	public int numLags = 0;
	
	ControlPanel(GeneralRepository repository) {
		mRepository = repository;
		
		mRepository.getPublisher().registerObserver(this);
		
		//crosshairDialog = new CrosshairPropertiesDialog(this, mRepository);
		//labelDialog = new LabelPropertiesDialog(mRepository);
		
//		numLags = repository.getGeneral().getWindowSize();
		
		setAlignmentX(JPanel.LEFT_ALIGNMENT);
		
		setBorder(new LineBorder(Color.BLACK, 1));
		layOutTheWidgets();

		numRowsField.addKeyListener(this);
		firstSliceField.addKeyListener(this);
		stepField.addKeyListener(this);
		lastSliceField.addKeyListener(this);
		sliceInputField.addKeyListener(this);
		lagsField.addKeyListener(this);
		
		crosshairButton.addActionListener(this);
		labelButton.addActionListener(this);
		sliceInstructionButton1.addActionListener(this);
		sliceInstructionButton2.addActionListener(this);
		plotButton.addActionListener(this);
		resetButton.addActionListener(this);
		allLagsCheckBox.addActionListener(this);
	}
	
	private void layOutTheWidgets() {
		this.setLayout(new BoxLayout(this,BoxLayout.PAGE_AXIS));
		
		JPanel propertiesPanel = new JPanel();
		propertiesPanel.setLayout(new BoxLayout(propertiesPanel, BoxLayout.X_AXIS));
		propertiesPanel.add(crosshairButton);
		propertiesPanel.add(labelButton);
		
		JPanel numRowsPanel = new JPanel();
		numRowsPanel.add(new JLabel("Max number of rows per lag: "));
		numRowsPanel.add(numRowsField);
		
		JPanel sliceInstructionPanel1 = new JPanel();
		sliceInstructionPanel1.add(sliceInstructionButton1);
		
		JPanel firstSlicePanel = new JPanel();
		firstSlicePanel.add(firstSliceLabel);
		firstSlicePanel.add(firstSliceField);
		
		JPanel stepPanel = new JPanel();
		stepLabel.setPreferredSize(firstSliceLabel.getPreferredSize());
		stepPanel.add(stepLabel);
		stepPanel.add(stepField);
		
		JPanel lastSlicePanel = new JPanel();
		lastSlicePanel.add(lastSliceLabel);
		lastSlicePanel.add(lastSliceField);
		
		JPanel sliceInstructionPanel2 = new JPanel();
		sliceInstructionPanel2.add(sliceInstructionButton2);
		
		JPanel sliceInputExamplePanel = new JPanel();
		sliceInputExampleLabel.setEnabled(false);
		sliceInputExamplePanel.add(sliceInputExampleLabel);
		
		JPanel sliceInputPanel = new JPanel();
		sliceInputField.setEnabled(false);
		sliceInputPanel.add(sliceInputField);
		
		JPanel lagsPanel = new JPanel(new BorderLayout());
		JPanel displayAllPanel = new JPanel();
		JPanel manualSelectPanel = new JPanel();
		
		displayAllPanel.add(displayAllLagsLabel);
		displayAllPanel.add(allLagsCheckBox);
		manualSelectPanel.add(new JLabel("Select lags by manual input: "));
		manualSelectPanel.add(lagsField);
		lagsPanel.add(displayAllPanel, BorderLayout.NORTH);
		lagsPanel.add(manualSelectPanel, BorderLayout.SOUTH);

		//Displaying all lags on startup is default behavior
		allLagsCheckBox.setSelected(true);
		lagsField.setEnabled(false);

		JPanel plotButtonPanel = new JPanel();
		plotButtonPanel.add(plotButton);
		plotButtonPanel.add(resetButton);
		
		// Adds the two buttons to a button group so only one of them
		// can be selected at a time.
		ButtonGroup buttonGroup = new ButtonGroup();
		buttonGroup.add(sliceInstructionButton1);
		buttonGroup.add(sliceInstructionButton2);

		this.add(propertiesPanel);
		this.add(new JSeparator());
		this.add(numRowsPanel);
		this.add(sliceInstructionPanel1);
		this.add(firstSlicePanel);
		this.add(stepPanel);
		this.add(lastSlicePanel);
		this.add(sliceInstructionPanel2);
		this.add(sliceInputExamplePanel);
		this.add(sliceInputPanel);
		this.add(lagsPanel);
		
		updateLagBox();
		updateSliceBox();
		numRowsField.setText("1");
		
		this.add(plotButtonPanel);
		
		restoreInputFields(BrainData.AXIAL);
	}
	
	// Helper method that stores the values for the first slice, step, last
	// slice, and the slice range. This is done in case the view is switched
	// and the slices have to be redrawn.
	protected void storeInputFields(int brainView) {
		sliceInstructionButton1Selected[brainView] = sliceInstructionButton1.isSelected();
	}
	
	// Helper method that restores the input fields related to the brain
	// slices, in case inputs were previously given.
	protected void restoreInputFields(int brainView) {
		sliceInstructionButton1.setSelected(sliceInstructionButton1Selected[brainView]);
		sliceInstructionButton2.setSelected(!sliceInstructionButton1Selected[brainView]);
		updateWidgets(sliceInstructionButton1Selected[brainView]);
	}
	
	// Helper method that resets the input fields related to the brain slices.
	protected void resetInputFields(int brainView) {
		firstSliceField.setText(ONE);
		stepField.setText(ONE);
		
		int numLags = mRepository.getGeneral().getWindowSize();
		
		lastSliceField.setText(Integer.toString(
				mRepository.getGeneral().getBrainData().getNumSlices(brainView)));
		
		sliceInputField.setText("");
		numRowsField.setText(ONE);
		lagsField.setText(ZERO + "-" + Integer.toString(numLags - 1));
		
		// Selects the first slice selection button as that is the default one.
		sliceInstructionButton1.setSelected(true);
		sliceInstructionButton1Selected[brainView] = true;

		allLagsCheckBox.setSelected(true);
		lagsField.setEnabled(false);
	}

	public void updateBrainView() {
		restoreInputFields(mRepository.getImagePropertiesModel().getBrainView() );
	}
		
	// The action listener functions (the combo boxes would only work
	// with these.. mouseListener doesn't work properly
	public void actionPerformed(ActionEvent e) {
		int numLags = mRepository.getGeneral().getWindowSize();
		
		if (e.getSource() == sliceInstructionButton1) {
			updateWidgets(true);
			
		} else if (e.getSource() == sliceInstructionButton2) {
			updateWidgets(false);
			
		} else if (e.getSource() == plotButton) {
			plotButtonAction();
			
		} else if (e.getSource() == resetButton) {
			int brainView = mRepository.getImagePropertiesModel().getBrainView();
			resetInputFields(brainView);
			storeInputFields(brainView);
			updateWidgets(true);
			
			// Redraws the brain latent variables plot to show all the
			// lags and slices again, with one row only for each lag.
			int numSlices = getNumSlices(brainView);
			ArrayList<Integer> lags = new ArrayList<Integer>();
			for (int i = 0; i < numLags; ++i) {
				lags.add(i);
			}
			ArrayList<Integer> slices = new ArrayList<Integer>();
			for (int i = 1; i <= numSlices; ++i) {
				slices.add(i);
			}
			ResultsCommandManager.setFilters(brainView, lags, slices, 1, true);
			
		} else if (e.getSource() == crosshairButton) {
			if (crosshairDialog == null) {
				crosshairDialog = new CrosshairPropertiesDialog(this, mRepository);
			}
			
			if (!crosshairDialog.isVisible()) {
				crosshairDialog.setVisible(true);
				crosshairDialog.setResizable(false);
			}
		} else if (e.getSource() == labelButton) {
			if (labelDialog == null) {
				labelDialog = new LabelPropertiesDialog(this, mRepository);
			}
			
			if (!labelDialog.isVisible()) {
				labelDialog.setVisible(true);
				labelDialog.setResizable(false);
			}
			
		} else if (e.getSource() == allLagsCheckBox){
			int brainView = mRepository.getImagePropertiesModel().getBrainView();
			
			if(allLagsCheckBox.isSelected()){
				lagsField.setEnabled(false);
			}else{
				lagsField.setEnabled(true);
			}
			updateLagBox();
		}
	}
    
	public void keyPressed(KeyEvent e) {
		
		// We only handle the case where the enter key is pressed.
		if (e.getKeyChar() != ControlPanel.ENTER_KEY) {
			return;
		}
		
		Object source = e.getSource();
		if (source == firstSliceField || source == stepField
			|| source == lastSliceField || source == sliceInputField
			|| source == numRowsField || source == lagsField) {
			
			plotButtonAction();
    	}
	}
	
	// Helper method that is called when the PLOT button is invoked or when
	// the enter key is pressed on any of the input fields relating to
	// slice values.
	private void plotButtonAction() {
		int numLags = mRepository.largestWindowSize();
		int brainView = mRepository.getImagePropertiesModel().getBrainView();
		boolean displayAll = allLagsCheckBox.isSelected();

		// Here, the fields are checked in the following order: lag selection, 
		// slice selection, and number of rows per lag.
		ArrayList<Integer> lagNumbers = null;
		ArrayList<Integer> sliceNumbers;
		int numRows;

		//Are we manually selecting certain lags? If so we need to generate a set
		//otherwise lagNumbers can be null since we will be displaying all the
		//lags.
		if(!displayAll){
			String lagRange = lagsField.getText().trim();

			if (!isLagOrSliceRangeValid(ControlPanel.LAG, lagRange, numLags - 1)) {
				return;
			}
			lagNumbers = getLagOrSliceNumbers(lagRange);
		}

		// Retrieves the number of slices, depending on the brain
		// view.
		int numSlices = getNumSlices(brainView);
		
		if (sliceInstructionButton1.isSelected()) {
			String firstSliceText = firstSliceField.getText().trim();
			String stepText = stepField.getText().trim();
			String lastSliceText = lastSliceField.getText().trim();
		
			// Checks if all the required fields were filled in first.
			if (firstSliceText.equals("")) {
				GlobalVariablesFunctions.showErrorMessage("No value was given for the first slice.");
				return;
			} else if (stepText.equals("")) {
				GlobalVariablesFunctions.showErrorMessage("No step value was given.");
				return;
			} else if (lastSliceText.equals("")) {
				GlobalVariablesFunctions.showErrorMessage("No value was given for the last slice.");
				return;
			}
		
			// Checks if integers are given as the values next.
			try {
				int firstSlice = Integer.parseInt(firstSliceText);
				int step = Integer.parseInt(stepText);
				int lastSlice = Integer.parseInt(lastSliceText);
			
				// Checks if the given values are in the correct
				// range first.
				if (!areFirstLastLagsOrSlicesValid(ControlPanel.SLICE, firstSlice, lastSlice, numSlices)) {
					return;
				} else if (step < 1) {
					GlobalVariablesFunctions.showErrorMessage("The step value must be positive.");
					return;
				}
				
				// Retrieves the slice values.
				sliceNumbers = getSliceNumbers(firstSlice, step, lastSlice);
			} catch (NumberFormatException ex) {
				GlobalVariablesFunctions.showErrorMessage("Only integers are allowed as the slice values.");
				return;
			}
			
		} else {
			String sliceRange = sliceInputField.getText().trim();
			
			if (!isLagOrSliceRangeValid(ControlPanel.SLICE, sliceRange, numSlices)) {
				return;
			}
			sliceNumbers = getLagOrSliceNumbers(sliceRange);
		}
		
		// Next, checks if the field for the number of lags per row is filled in.
		String numRowsText = numRowsField.getText().trim();
		if (numRowsText.equals("")) {
			GlobalVariablesFunctions.showErrorMessage("No value was given for the number of rows per lag.");
			return;
		}
		
		// Checks if the given number of rows per lag is a valid value.
		try {
			numRows = Integer.parseInt(numRowsText);
			if (numRows < 1 || numRows > numSlices) {
				GlobalVariablesFunctions.showErrorMessage("The number of rows per lag must be between 1 and " + numSlices + ".");
				return;
			}
		} catch (NumberFormatException e) {
			GlobalVariablesFunctions.showErrorMessage("Only an integer is allowed as the number of rows per lag.");
			return;
		}
		
		// Stores the values from the input text fields only when the user
		// presses the PLOT button or the enter key.
		storeInputFields(brainView);
		
		// Redraws the brain latent variables plot to only show the
		// selected lags and slices.
		ResultsCommandManager.setFilters(brainView, lagNumbers,
				sliceNumbers, numRows, displayAll);
	}
	
	// Helper method for returning the total number of slices in the given
	// brain view.
	private int getNumSlices(int brainView) {
		return mRepository.getGeneral().getBrainData().getNumSlices(brainView);
	}
	
	// Helper method that checks whether the given range is valid based
	// on the given type (either "lag" or "slice") and the total number
	// of the given type.
	private boolean isLagOrSliceRangeValid(String type, String range, int numTypes) {
		
		// Checks if the required field was filled in first.
		if (range.equals("")) {
			GlobalVariablesFunctions.showErrorMessage("No " + type + " range was given.");
			return false;
			
		// Otherwise, checks if valid values were given as the range.
			
		// Checks if the given range is just a number.
		} else if (range.matches(ControlPanel.NUMBER_REGEX)) {
			int typeNum = Integer.parseInt(range);
			if (!isLagOrSliceValid(type, typeNum, numTypes, false)) {
				return false;
			}
			
		// Checks if the given range is of the form a-b. Spaces are
		// allowed before and after the hyphen (-).
		} else if (range.matches(ControlPanel.HYPHEN_RANGE_REGEX)) {
			String[] newRange = range.split("-");
			int first = Integer.parseInt(newRange[0].trim());
			int last = Integer.parseInt(newRange[1].trim());
			if (!areFirstLastLagsOrSlicesValid(type, first, last, numTypes)) {
				return false;
			}
		
		// Checks if the given range is of the form a, b, c. Spaces are
		// allowed before and after the commas.
		} else if (range.matches(ControlPanel.COMMA_RANGE_REGEX)) {
			String[] newRange = range.split(",");
			for (int i = 0; i != newRange.length; i++) {
				int typeNum = Integer.parseInt(newRange[i].trim());
				if (!isLagOrSliceValid(type, typeNum, numTypes, true)) {
					return false;
				}
			}
		
		// Checks if the given range is a combination of the two cases
		// above.
		} else if (range.matches(ControlPanel.MULTI_RANGE_REGEX)) {
			String[] newRange = range.split(",");
			for (int i = 0; i != newRange.length; i++) {
				String rangeItem = newRange[i].trim();
				if (rangeItem.indexOf('-') != -1) {
					String[] miniRange = rangeItem.split("-");
					int first = Integer.parseInt(miniRange[0].trim());
					int last = Integer.parseInt(miniRange[1].trim());
					if (!areFirstLastLagsOrSlicesValid(type, first, last, numTypes)) {
						return false;
					}
				} else {
					int typeNum = Integer.parseInt(rangeItem);
					if (!isLagOrSliceValid(type, typeNum, numTypes, true)) {
						return false;
					}
				}
			}
			
		} else {
			GlobalVariablesFunctions.showErrorMessage("Invalid values were given as the " + type + " range.");
			return false;
		}
		
		return true;
	}



	/**
	 * Helper method that checks if the given lag or slice number is valid based on
	 * the number of slices.
	 * @param type LAG or SLICE
	 * @param typeNum the slice/lag number
	 * @param numTypes (in the case of lags, the number of lags available i.e
	 * the windowsize)
	 * @param multipleTypesInvolved This is a variable to modify the error given
	 * to the user. If we are checking values 'a,b,c' and one is wrong we say
	 * something different than if a single value 'a' is wrong.
	 * @return true if the slice/lag is valid, false otherwise
	 */
	private boolean isLagOrSliceValid(String type, int typeNum, 
									  int numTypes, boolean multipleTypesInvolved) {
		int min;
		
		// The minimum value for a lag is 0 while the minimum value
		// for a slice is 1.
		if (type.equals(ControlPanel.LAG)) {
			min = 0;
		} else {
			min = 1;
		}
		
		if ((typeNum < min || typeNum > numTypes) && multipleTypesInvolved) {
			GlobalVariablesFunctions.showErrorMessage("All given " + type + "s must be between " + min + " and " + numTypes + ".");
		} else if (typeNum < min || typeNum > numTypes) {
			GlobalVariablesFunctions.showErrorMessage("The given " + type + " must be between " + min + " and " + numTypes + ".");
		} else {
			return true;
		}
		
		return false;
	}
	
	// Helper method that checks if the given first and last lag or slice values
	// are valid based on the total number given.
	private boolean areFirstLastLagsOrSlicesValid(String type, int firstType, int lastType, int numTypes) {
		int min;
		
		// The minimum value for a lag is 0 while the minimum value
		// for a slice is 1.
		if (type.equals(ControlPanel.LAG)) {
			min = 0;
		} else {
			min = 1;
		}
		
		if (firstType < min || firstType > numTypes) {
			GlobalVariablesFunctions.showErrorMessage("The first " + type + " must be between " + min + " and " + numTypes + ".");
		} else if (lastType < min || lastType > numTypes) {
			GlobalVariablesFunctions.showErrorMessage("The last " + type + " must be between " + min + " and " + numTypes + ".");
		} else if (lastType < firstType) {
			GlobalVariablesFunctions.showErrorMessage("The first " + type + " should not be larger than the last " + type + ".");
		} else {
			return true;
		}
		
		return false;
	}
	
	// Retrieves the slice numbers based on the given values of the first slice, the
	// step, and the last slice. It is assumed here that the given slice values are
	// valid, depending on the current brain view.
	private ArrayList<Integer> getSliceNumbers(int firstSlice, int step, int lastSlice) {
		ArrayList<Integer> sliceNumbers = new ArrayList<Integer>();
		
		int i = firstSlice;
		while (i <= lastSlice) {
			sliceNumbers.add(i);
			i += step;
		}
		
		return sliceNumbers;
	}
	
	// Retrieves the lag or slice numbers based on the given range. It is assumed
	// here that the given range contains valid slice values depending
	// on the current brain view.
	private ArrayList<Integer> getLagOrSliceNumbers(String sliceRange) {
		ArrayList<Integer> numbers = new ArrayList<Integer>();
		
		if (sliceRange.matches(ControlPanel.NUMBER_REGEX)) {
			int sliceNum = Integer.parseInt(sliceRange);
			numbers.add(sliceNum);
		} else if (sliceRange.matches(ControlPanel.HYPHEN_RANGE_REGEX)) {
			String[] range = sliceRange.split("-");
			int firstSlice = Integer.parseInt(range[0].trim());
			int lastSlice = Integer.parseInt(range[1].trim());
			for (int i = firstSlice; i <= lastSlice; i++) {
				numbers.add(i);
			}
		} else if (sliceRange.matches(ControlPanel.COMMA_RANGE_REGEX)) {
			String[] range = sliceRange.split(",");
			for (int i = 0; i != range.length; i++) {
				int sliceNum = Integer.parseInt(range[i].trim());
				if (!numbers.contains(sliceNum)) {
					numbers.add(sliceNum);
				}
			}
		} else if (sliceRange.matches(ControlPanel.MULTI_RANGE_REGEX)) {
			String[] range = sliceRange.split(",");
			for (int i = 0; i != range.length; i++) {
				String rangeItem = range[i].trim();
				if (rangeItem.indexOf('-') != -1) {
					String[] miniRange = rangeItem.split("-");
					int firstSlice = Integer.parseInt(miniRange[0].trim());
					int lastSlice = Integer.parseInt(miniRange[1].trim());
					for (int j = firstSlice; j <= lastSlice; j++) {
						if (!numbers.contains(j)) {
							numbers.add(j);
						}
					}
				} else {
					int sliceNum = Integer.parseInt(rangeItem);
					if (!numbers.contains(sliceNum)) {
						numbers.add(sliceNum);
					}
				}
			}
		}
			
		return numbers;
	}
	
	// Updates the widgets related to selecting slices, depending on if
	// the first slice instruction button is selected or not.
	private void updateWidgets(boolean sliceInstructionButton1Selected) {
		firstSliceLabel.setEnabled(sliceInstructionButton1Selected);
		firstSliceField.setEnabled(sliceInstructionButton1Selected);
		stepLabel.setEnabled(sliceInstructionButton1Selected);
		stepField.setEnabled(sliceInstructionButton1Selected);
		lastSliceLabel.setEnabled(sliceInstructionButton1Selected);
		lastSliceField.setEnabled(sliceInstructionButton1Selected);
		
		sliceInputExampleLabel.setEnabled(!sliceInstructionButton1Selected);
		sliceInputField.setEnabled(!sliceInstructionButton1Selected);
		
		int brainView = mRepository.getImagePropertiesModel().getBrainView();
		boolean displayAll = mRepository.getControlPanelModel().
				getAllLagsFlag(brainView);
		allLagsCheckBox.setSelected(displayAll);

		updateLagBox();
		updateSliceBox();
		updateNumRowsBox();
	}
	
	public void keyTyped(KeyEvent e) {}
	public void keyReleased(KeyEvent e) {}
	
	private String parseRange(ArrayList<Integer> someNumbers) {
		int start = -1;
		int end = -1;
		
		String rangeString = "";
		
		for (Integer lag : someNumbers) {
			if (start == -1) {
				start = lag;
				end = lag;
			}
			else if (lag == end + 1) {
				end = lag;
			}
			else {
				if (start != end) {
					rangeString += start + "-" + end + ",";
				}
				else {
					rangeString += start + ",";
				}
				
				start = lag;
				end = lag;
			}
		}
		
		if (start != -1) {
			if (start != end) {
				rangeString += start + "-" + end;
			}
			else {
				rangeString += start;
			}
		}
		
		return rangeString;
	}
	
	private void updateSliceBox() {
		
		int brainView = mRepository.getImagePropertiesModel().getBrainView();
		ArrayList<Integer> sliceNums = mRepository.getControlPanelModel().getSliceNumbers(brainView);
		
		if (sliceInstructionButton1Selected[brainView]) {
			Integer firstSlice = sliceNums.get(0);
			Integer step = 1;
			if (sliceNums.size() > 1) {
				step = sliceNums.get(1) - sliceNums.get(0);
			}
			Integer lastSlice = sliceNums.get(sliceNums.size() - 1);
			
			firstSliceField.setText(firstSlice.toString() );
			stepField.setText(step.toString() );
			lastSliceField.setText(lastSlice.toString() );
		}
		else {
			String sliceString = parseRange(sliceNums);
			sliceInputField.setText(sliceString);
		}
	}

	/**Updates the lag filter box.
	 *
	 */
	private void updateLagBox() {
		int brainView = mRepository.getImagePropertiesModel().getBrainView();
		
		if(allLagsCheckBox.isSelected()){
			//allLagsCheckBox.setSelected(true);
			lagsField.setEnabled(false);
		}else{
			ArrayList<Integer> lags = mRepository.getControlPanelModel()
					.getLagNumbers(brainView);

			String lagString = parseRange(lags);
			lagsField.setText(lagString);
			lagsField.setEnabled(true);
			//allLagsCheckBox.setSelected(false);
		}
	}
	
	private void updateNumRowsBox() {
		int brainView = mRepository.getImagePropertiesModel().getBrainView();
		
		int numRows = mRepository.getControlPanelModel().getNumRowsPerLag(brainView);
		
		numRowsField.setText(Integer.toString(numRows) );
	}
    
	public void volumesAdded() {
//		updateLagBox();
//		updateSliceBox();
		restoreInputFields(mRepository.getImagePropertiesModel().getBrainView() );
	}

	/**
	 * Generates a warning dialog when a user selects lags to display that
	 * do not exist for certain files. This can happen when the user hits "PLOT"
	 * or when a new file is added that does not have the lags the user has
	 * specifically chosen.
	 * @param e The event that caused the warning (contains list of files and
	 * offending lags).
	 */
	private void generateIncorrectLagsWarning(IncorrectLagsSelectedEvent e){

		//Each file in here is guaranteed to have at least one warning.
		HashMap<String, ArrayList<ArrayList<Integer>>> warnings = e.getWarnings();
		String compiledWarnings = "The following selected lags do not ";
		compiledWarnings += "exist for the following files\n\n";
		
		int i;
		//used to avoid inserting a space at the beginning of printing out lags.
		boolean started = false;

		for(String file : warnings.keySet()){
			compiledWarnings += file + "\n";

			ArrayList<ArrayList<Integer>> bv = warnings.get(file);
			i = 0;
			
			for(ArrayList<Integer> invalidLags : bv){
				//Don't print out brainviews that have no offending lags.
				if(invalidLags.size() == 0){
					i++;
					continue;
				}

				if(started == true){
					compiledWarnings += " ";
				}
				
				switch(i){
					case 0:
						compiledWarnings += "AXIAL:";
						started = true;
						break;
					case 1:
						compiledWarnings += "SAGITTAL:";
						started = true;
						break;
					case 2:
						compiledWarnings += "CORONAL:";
						break;
				}
				compiledWarnings += " " + invalidLags.get(0).intValue();
				for(int j = 1; j< invalidLags.size(); j++){
					int invalidLag = invalidLags.get(j).intValue();
					compiledWarnings += "," + invalidLag;
				}
				i++;
			}
			compiledWarnings += "\n\n";
		}
		JOptionPane.showMessageDialog(null, compiledWarnings);
	}
	
	@Override
	public void notify(SliceFiltersEvent e) {
		updateBrainView();
	}

	@Override
	public void notify(ViewedLvsEvent e) {}

	@Override
	public void notify(Event e) {}

	@Override
	public void notify(BrainFilterEvent e) {
		// TODO Auto-generated method stub
	}

	public void notify(IncorrectLagsSelectedEvent e){
		generateIncorrectLagsWarning(e);
	}
}