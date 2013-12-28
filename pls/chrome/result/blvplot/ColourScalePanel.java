package pls.chrome.result.blvplot;

import java.awt.Color;
import java.awt.Font;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;

import javax.swing.BoxLayout;
import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.border.LineBorder;

import pls.chrome.result.controller.ResultsCommandManager;
import pls.chrome.result.controller.command.ToggleUseGlobalColourScaleCommand;
import pls.chrome.result.model.GeneralRepository;

@SuppressWarnings("serial")
public class ColourScalePanel extends JPanel implements ActionListener, KeyListener {
	private static int BORDER_THICKNESS = 1;
	
	private JButton mSetButton = new JButton("Set");
	private JButton mResetButton = new JButton("Reset");
	private MaxMinThreshPanel mInputPanel = new MaxMinThreshPanel();
	private JRadioButton mUseGlobal = new JRadioButton("Use a global colour scale", true);
	private JRadioButton mUseIndividual = new JRadioButton("Use individual colour scales", true);
	
	private GeneralRepository mRepository = null;
	
	public static final int noColourButtonHasBeenPressed = 0;
	public static final int globalColourButtonLastPressed = 1;
	public static final int individualColourButtonLastPressed = 2;
	private int lastColourButtonPressed = noColourButtonHasBeenPressed;
	
	public ColourScalePanel(GeneralRepository repository) {
		mRepository = repository;
		
		setAlignmentX(JPanel.LEFT_ALIGNMENT);
		setBorder(new LineBorder(Color.DARK_GRAY, BORDER_THICKNESS) );
		setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
		
		JLabel title = new JLabel("Global colour scale");
		Font font = title.getFont();
		// Make the title have a bold and slightly larger font
		title.setFont(new Font(font.getFontName(), Font.BOLD, font.getSize() + 1));
		
//		add(title);
		
		JPanel radioPanel = new JPanel();
		radioPanel.setAlignmentX(JPanel.LEFT_ALIGNMENT);
		ButtonGroup bg = new ButtonGroup();
		
		bg.add(mUseGlobal);
		radioPanel.add(mUseGlobal);
		bg.add(mUseIndividual);
		radioPanel.add(mUseIndividual);
		add(radioPanel);
		
		add(mInputPanel);
		
		// Buttons
		JPanel panel = new JPanel();
		panel.setLayout(new BoxLayout(panel, BoxLayout.X_AXIS));
		panel.setAlignmentX(JPanel.LEFT_ALIGNMENT);
		panel.add(mSetButton);
		panel.add(mResetButton);
		add(panel);
		
		updateColourScale();
		
		mUseGlobal.addActionListener(this);
		mUseIndividual.addActionListener(this);
		mSetButton.addActionListener(this);
		mResetButton.addActionListener(this);
		
		mInputPanel.mMaxTextField.addKeyListener(this);
		mInputPanel.mMinTextField.addKeyListener(this);
		mInputPanel.mThreshTextField.addKeyListener(this);
	}
	
	public void actionPerformed(ActionEvent event) {
		if (event.getSource() == mSetButton)
		{
			issueColourScaleCommand();
		}
		else if (event.getSource() == mResetButton) {
			double[] colourScaleNumbers = null;
			
			if (mRepository.getUseGlobalScale() ) {
				colourScaleNumbers = mRepository.getCalculatedColourScale();
			}
			else {
				colourScaleNumbers = mRepository.getGeneral().getBrainData().getMaxMinThresh();
			}
			
			mInputPanel.setColourScale(colourScaleNumbers);
			issueColourScaleCommand();
		}
		else if (event.getSource() == mUseGlobal) {
			if (lastColourButtonPressed == noColourButtonHasBeenPressed || lastColourButtonPressed == individualColourButtonLastPressed) {
				ResultsCommandManager.setUseGlobalScale();
				lastColourButtonPressed = globalColourButtonLastPressed;
				}
			
		}
		else if (event.getSource() == mUseIndividual) {
			if (lastColourButtonPressed == noColourButtonHasBeenPressed || lastColourButtonPressed == globalColourButtonLastPressed) {
				ResultsCommandManager.setUseGlobalScale();
				lastColourButtonPressed = individualColourButtonLastPressed;
				}	
		}
	}
	
	/**
	 * Executes a colour scale command based on the current values in the max,
	 * min and threshold textfields.
	 **/
	private void issueColourScaleCommand() {
		double max;
		double min;
		double thresh;
		
		double[] colours = mInputPanel.getColourScale();
		
		max = colours[0];
		min = colours[1];
		thresh = colours[2];
		
		if (mRepository.getUseGlobalScale() ) {
			ResultsCommandManager.setGlobalColourScale(max, min, thresh);
		}
		else {
			ResultsCommandManager.setColourScale(max, min, thresh);
		}
		
		repaint();
	}

	public void keyPressed(KeyEvent e) {
		// We only handle the case where the enter key has been pressed.
		if (e.getKeyChar() != BrainLatentVariablesPlot.ENTER_KEY) {
			return;
		}
		
		if (e.getSource() == mInputPanel.mMaxTextField
				|| e.getSource() == mInputPanel.mMinTextField
				|| e.getSource() == mInputPanel.mThreshTextField) {

			issueColourScaleCommand();
		}
	}

	public void keyReleased(KeyEvent arg0) {
	}

	public void keyTyped(KeyEvent arg0) {
	}

	public void refreshWidgets() {
		updateColourScale();
	}
	
	public void updateColourScale() {
		double[] colourScale = null;
		
		if (mRepository.getUseGlobalScale() ) {
			colourScale = mRepository.getGlobalColourScale();
		}
		else {
			//get the currently selected model, get its braindata, then get its
			//colour scale.
			colourScale = mRepository.getGeneral().getBrainData()
					.getColourScaleModel().getColourScale();
		}
		
		mInputPanel.setColourScale(colourScale);
	}
}
