package pls.chrome.result.blvplot;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;

import javax.swing.BoxLayout;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JColorChooser;
import javax.swing.JComponent;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.SpringLayout;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import pls.chrome.result.controller.ResultsCommandManager;
import pls.chrome.result.controller.observer.Event;
import pls.chrome.result.controller.observer.brainimageproperties.BackgroundImageEvent;
import pls.chrome.result.controller.observer.brainimageproperties.BrainImagePropertiesObserver;
import pls.chrome.result.controller.observer.brainimageproperties.CrosshairColourEvent;
import pls.chrome.result.controller.observer.brainimageproperties.CrosshairTransparencyEvent;
import pls.chrome.result.controller.observer.brainimageproperties.LabelsColourEvent;
import pls.chrome.result.controller.observer.brainimageproperties.LabelsTransparencyEvent;
import pls.chrome.result.controller.observer.brainimageproperties.UseCrosshairEvent;
import pls.chrome.result.controller.observer.brainimageproperties.UseDescriptionLabelsEvent;
import pls.chrome.result.controller.observer.brainimageproperties.UseLabelsEvent;
import pls.chrome.result.model.GeneralRepository;

@SuppressWarnings("serial")
public class CrosshairPropertiesDialog extends JDialog
implements ActionListener, ChangeListener, BrainImagePropertiesObserver {

	// The default crosshair color is green.
	private Color crosshairColor = Color.GREEN;
	
	private static final int COLOR_BUTTON_LENGTH = 15;
	private Image colorImage = new BufferedImage(COLOR_BUTTON_LENGTH, COLOR_BUTTON_LENGTH, BufferedImage.TYPE_INT_RGB);
	
	private JCheckBox crosshairButton = new JCheckBox("Use crosshair", true);
	private JLabel crosshairTransparencyLabel = new JLabel("Set crosshair transparency:");
	private JSlider crosshairTransparencySlider = new JSlider(0, 255, 255);
	private JLabel crosshairColorLabel = new JLabel("Set crosshair colour:");
	private JButton crosshairColorButton = new JButton(new ImageIcon(colorImage));
	
	private JButton closeButton = new JButton("Close");
	
	private GeneralRepository mRepository;
	
	public CrosshairPropertiesDialog(JComponent parent, GeneralRepository repository) {
		super(JOptionPane.getFrameForComponent(parent) );
		
		mRepository = repository;
		mRepository.getPublisher().registerObserver(this);
		
		setTitle("Crosshair Properties");
		setSize(400, 200);
		setLayout(new BoxLayout(getContentPane(), BoxLayout.Y_AXIS));
		
		updateColorButton();
		
		JPanel crosshairPanel = new JPanel();
		SpringLayout s = new SpringLayout();
		crosshairPanel.setLayout(s);
		
		crosshairPanel.add(crosshairButton);
		crosshairPanel.add(crosshairTransparencyLabel);
		crosshairPanel.add(crosshairTransparencySlider);
		crosshairPanel.add(crosshairColorLabel);
		crosshairPanel.add(crosshairColorButton);
		
		s.putConstraint(SpringLayout.NORTH, crosshairButton, 10, SpringLayout.NORTH, crosshairPanel);
		s.putConstraint(SpringLayout.WEST, crosshairButton, 5, SpringLayout.WEST, crosshairPanel);
		s.putConstraint(SpringLayout.NORTH, crosshairTransparencyLabel, 10, SpringLayout.SOUTH, crosshairButton);
		s.putConstraint(SpringLayout.WEST, crosshairTransparencyLabel, 10, SpringLayout.WEST, crosshairPanel);
		s.putConstraint(SpringLayout.NORTH, crosshairTransparencySlider, 10, SpringLayout.SOUTH, crosshairButton);
		s.putConstraint(SpringLayout.WEST, crosshairTransparencySlider, 5, SpringLayout.EAST, crosshairTransparencyLabel);

		s.putConstraint(SpringLayout.NORTH, crosshairColorLabel, 10, SpringLayout.SOUTH, crosshairTransparencyLabel);
		s.putConstraint(SpringLayout.WEST, crosshairColorLabel, 10, SpringLayout.WEST, crosshairPanel);
		s.putConstraint(SpringLayout.NORTH, crosshairColorButton, 10, SpringLayout.SOUTH, crosshairTransparencyLabel);
		s.putConstraint(SpringLayout.WEST, crosshairColorButton, 5, SpringLayout.WEST, crosshairTransparencySlider);
		
		int width = crosshairTransparencyLabel.getPreferredSize().width
				  + crosshairTransparencySlider.getPreferredSize().width
				  + 15;
		int height = crosshairButton.getPreferredSize().height
				   + crosshairTransparencySlider.getPreferredSize().height
				   + crosshairColorButton.getPreferredSize().height
				   + 70;
		Dimension dimension = new Dimension(width, height);
		crosshairPanel.setPreferredSize(dimension);
		
		JPanel closePanel = new JPanel();
		closePanel.setLayout(new BoxLayout(closePanel, BoxLayout.X_AXIS));
		closePanel.setAlignmentX(JPanel.CENTER_ALIGNMENT);
		closePanel.add(closeButton);
		
		add(crosshairPanel);
		add(closePanel);
		
		crosshairTransparencySlider.addChangeListener(this);
		crosshairButton.addActionListener(this);
		crosshairColorButton.addActionListener(this);
		closeButton.addActionListener(this);
	}
	
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == crosshairButton) {
			boolean isSelected = crosshairButton.isSelected();
			crosshairTransparencyLabel.setEnabled(isSelected);
			crosshairTransparencySlider.setEnabled(isSelected);
			crosshairColorLabel.setEnabled(isSelected);
			crosshairColorButton.setEnabled(isSelected);
			ResultsCommandManager.toggleCrosshair();
			
		} else if (e.getSource() == crosshairColorButton) {
			Color color = JColorChooser.showDialog(null, "Select Crosshair Color", crosshairColor);
			if (color != null && !color.equals(crosshairColor)) {
				ResultsCommandManager.setCrosshairColor(color);
			}
			
		} else if (e.getSource() == closeButton) {
			setVisible(false);
		}
	}
	
	public void stateChanged(ChangeEvent e) {
		if (e.getSource() == crosshairTransparencySlider) {
			ResultsCommandManager.setCrosshairTransparency(crosshairTransparencySlider.getValue());
		}
	}
	
	private void updateColorButton() {
		crosshairColor = mRepository.getImagePropertiesModel().getCrosshairColour();
		Graphics2D g2 = (Graphics2D) colorImage.getGraphics();
		g2.setPaint(crosshairColor);
		g2.fillRect(0, 0, COLOR_BUTTON_LENGTH, COLOR_BUTTON_LENGTH);
		
		repaint();
	}
	
	private void updateTransparencySlide() {
		int transparencyLevel = mRepository.getImagePropertiesModel().getCrosshairTransparency();
		
		crosshairTransparencySlider.removeChangeListener(this);
		
		crosshairTransparencySlider.setValue(transparencyLevel);
		
		crosshairTransparencySlider.addChangeListener(this);
	}
	
	public void toggleUseCrosshair() {
		boolean isEnabled = mRepository.getImagePropertiesModel().isCrosshairEnabled();
		
		
		if (crosshairButton.isSelected() != isEnabled) {
			crosshairButton.removeActionListener(this);
			crosshairButton.setSelected(isEnabled);
			crosshairButton.addActionListener(this);
		}
	}

	///////////////////////////////////////////////////////////////////////////
	// Event handling
	///////////////////////////////////////////////////////////////////////////
	@Override
	public void notify(BackgroundImageEvent e) {}

	///////////////////////////////////////////////////////////////////////////
	// Brain image property event handlers
	///////////////////////////////////////////////////////////////////////////
	@Override
	public void notify(CrosshairColourEvent e) {
		updateColorButton();
	}

	@Override
	public void notify(CrosshairTransparencyEvent e) {
		updateTransparencySlide();
	}

	@Override
	public void notify(LabelsColourEvent e) {}

	@Override
	public void notify(LabelsTransparencyEvent e) {}

	@Override
	public void notify(UseCrosshairEvent e) {
		toggleUseCrosshair();
	}

	@Override
	public void notify(UseDescriptionLabelsEvent e) {}

	@Override
	public void notify(UseLabelsEvent e) {}

	@Override
	public void notify(Event e) {}
}