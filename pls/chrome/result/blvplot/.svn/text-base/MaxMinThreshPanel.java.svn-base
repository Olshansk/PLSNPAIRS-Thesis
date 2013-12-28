package pls.chrome.result.blvplot;

import javax.swing.BoxLayout;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;

import pls.shared.GlobalVariablesFunctions;

@SuppressWarnings("serial")
public class MaxMinThreshPanel extends JPanel {
	protected JTextField mMaxTextField = new JTextField(8);
	protected JTextField mMinTextField = new JTextField(8);
	protected JTextField mThreshTextField = new JTextField(8);

	public MaxMinThreshPanel() {
		initializeWidgets();
	}
	
	private void initializeWidgets() {
		setAlignmentX(JPanel.LEFT_ALIGNMENT);
		setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
		
		JLabel maxLabel = new JLabel("Maximum: ");
		JLabel minLabel = new JLabel("Minimum: ");
		JLabel threshLabel = new JLabel("Threshold: ");
		
		// Max
//		add(maxLabel);
		JPanel panel = new JPanel();
		panel.setLayout(new BoxLayout(panel, BoxLayout.X_AXIS));
		panel.setAlignmentX(JPanel.LEFT_ALIGNMENT);
		panel.add(maxLabel);
		panel.add(mMaxTextField);
//		panel.add(mMaxResetButton);
		add(panel);

		// Min
//		add(minLabel);
		panel = new JPanel();
		panel.setLayout(new BoxLayout(panel, BoxLayout.X_AXIS));
		panel.setAlignmentX(JPanel.LEFT_ALIGNMENT);
		panel.add(minLabel);
		panel.add(mMinTextField);
//		panel.add(mMinResetButton);
		add(panel);
		
		// Threshold
//		add(threshLabel);
		panel = new JPanel();
		panel.setLayout(new BoxLayout(panel, BoxLayout.X_AXIS));
		panel.setAlignmentX(JPanel.LEFT_ALIGNMENT);
		panel.add(threshLabel);
		panel.add(mThreshTextField);
//		panel.add(mThreshResetButton);
		add(panel);
	}

	public void setColourScale(double[] colourScaleNumbers) {
		mMaxTextField.setText(Double.toString(colourScaleNumbers[0]) );
		mMinTextField.setText(Double.toString(colourScaleNumbers[1]) );
		mThreshTextField.setText(Double.toString(colourScaleNumbers[2]) );
	}
	
	public double[] getColourScale() {
		double max;
		double min;
		double thresh;
		
		try {
			max = Double.parseDouble(mMaxTextField.getText() );
		}
		catch (NumberFormatException e) {
			GlobalVariablesFunctions.showErrorMessage("The maximum value for the colour scale must be a number.", "Incorrect value");
			mMaxTextField.requestFocus();
			return null;
		}
		
		try {
			min = Double.parseDouble(mMinTextField.getText() );
		}
		catch (NumberFormatException e) {
			GlobalVariablesFunctions.showErrorMessage("The minimum value for the colour scale must be a number.", "Incorrect value");
			mMinTextField.requestFocus();
			return null;
		}
		
		try {
			thresh = Double.parseDouble(mThreshTextField.getText() );
		}
		catch (NumberFormatException e) {
			GlobalVariablesFunctions.showErrorMessage("The threshold for the colour scale must be a number.", "Incorrect value");
			mThreshTextField.requestFocus();
			return null;
		}
		
		return new double[]{max, min, thresh};
	}
	
	public void setEnabled(boolean enabled) {
		mMaxTextField.setEnabled(enabled);
		mMinTextField.setEnabled(enabled);
		mThreshTextField.setEnabled(enabled);
	}
}
