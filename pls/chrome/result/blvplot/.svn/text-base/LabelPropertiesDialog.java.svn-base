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
import pls.chrome.result.controller.observer.Observer;
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
public class LabelPropertiesDialog extends JDialog
implements ActionListener, ChangeListener, BrainImagePropertiesObserver {

	// The default label color is magenta.
	public Color labelColor = Color.MAGENTA;
	
	private static final int COLOR_BUTTON_LENGTH = 15;
	private Image colorImage = new BufferedImage(COLOR_BUTTON_LENGTH, COLOR_BUTTON_LENGTH, BufferedImage.TYPE_INT_RGB);
	
	private JCheckBox descriptionLabelButton = new JCheckBox("Show description labels", true);
	private JCheckBox labelButton = new JCheckBox("Use lag/slice labels", true);
	private JLabel labelTransparencyLabel = new JLabel("Set label transparency:");
	private JSlider labelTransparencySlider = new JSlider(0, 255, 255);
	private JLabel labelColorLabel = new JLabel("Set label colour:");
	private JButton labelColorButton = new JButton(new ImageIcon(colorImage));
	
	private JButton closeButton = new JButton("Close");
	
	private GeneralRepository mRepository;
	
	public LabelPropertiesDialog(JPanel parent, GeneralRepository repository) {
		super(JOptionPane.getFrameForComponent(parent) );
		
		mRepository = repository;
		mRepository.getPublisher().registerObserver(this);
		
		setTitle("Lag/Slice Label Properties");
		setLayout(new BoxLayout(getContentPane(), BoxLayout.Y_AXIS));
		
		updateColorButton();
		
		JPanel labelPanel = new JPanel();
		SpringLayout s = new SpringLayout();
		labelPanel.setLayout(s);
		
		labelPanel.add(descriptionLabelButton);
		labelPanel.add(labelButton);
		labelPanel.add(labelTransparencyLabel);
		labelPanel.add(labelTransparencySlider);
		labelPanel.add(labelColorLabel);
		labelPanel.add(labelColorButton);
		
		s.putConstraint(SpringLayout.NORTH, descriptionLabelButton, 10, SpringLayout.NORTH, labelPanel);
		s.putConstraint(SpringLayout.WEST, descriptionLabelButton, 5, SpringLayout.WEST, labelPanel);
		s.putConstraint(SpringLayout.NORTH, labelButton, 10, SpringLayout.SOUTH, descriptionLabelButton);
		s.putConstraint(SpringLayout.WEST, labelButton, 0, SpringLayout.WEST, descriptionLabelButton);
		s.putConstraint(SpringLayout.NORTH, labelTransparencyLabel, 10, SpringLayout.SOUTH, labelButton);
		s.putConstraint(SpringLayout.WEST, labelTransparencyLabel, 10, SpringLayout.WEST, labelPanel);
		s.putConstraint(SpringLayout.NORTH, labelTransparencySlider, 10, SpringLayout.SOUTH, labelButton);
		s.putConstraint(SpringLayout.WEST, labelTransparencySlider, 5, SpringLayout.EAST, labelTransparencyLabel);

		s.putConstraint(SpringLayout.NORTH, labelColorLabel, 10, SpringLayout.SOUTH, labelTransparencyLabel);
		s.putConstraint(SpringLayout.WEST, labelColorLabel, 10, SpringLayout.WEST, labelPanel);
		s.putConstraint(SpringLayout.NORTH, labelColorButton, 10, SpringLayout.SOUTH, labelTransparencyLabel);
		s.putConstraint(SpringLayout.WEST, labelColorButton, 5, SpringLayout.WEST, labelTransparencySlider);
		
		int width = labelTransparencyLabel.getPreferredSize().width
				  + labelTransparencySlider.getPreferredSize().width
				  + 15;
		int height = descriptionLabelButton.getPreferredSize().height
					+ labelButton.getPreferredSize().height
					+ labelTransparencySlider.getPreferredSize().height
					+ labelColorButton.getPreferredSize().height
					+ 70;
		Dimension dimension = new Dimension(width, height);
		labelPanel.setPreferredSize(dimension);
		
		
		JPanel closePanel = new JPanel();
		closePanel.setLayout(new BoxLayout(closePanel, BoxLayout.X_AXIS));
		closePanel.setAlignmentX(JPanel.CENTER_ALIGNMENT);
		closePanel.add(closeButton);
		
		setSize(width + 10, height + closePanel.getPreferredSize().height);
		
		add(labelPanel);
		add(closePanel);
		
		descriptionLabelButton.addActionListener(this);
		labelTransparencySlider.addChangeListener(this);
		labelButton.addActionListener(this);
		labelColorButton.addActionListener(this);
		closeButton.addActionListener(this);
	}
	
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == labelButton) {
			boolean isSelected = labelButton.isSelected();
			labelTransparencyLabel.setEnabled(isSelected);
			labelTransparencySlider.setEnabled(isSelected);
			labelColorLabel.setEnabled(isSelected);
			labelColorButton.setEnabled(isSelected);
			ResultsCommandManager.toggleLabels();
			
		} else if (e.getSource() == descriptionLabelButton) {
			boolean isSelected = descriptionLabelButton.isSelected();
			ResultsCommandManager.toggleDescriptionLabels(isSelected);
		}
		else if (e.getSource() == labelColorButton) {
			Color color = JColorChooser.showDialog(null, "Select Label Color", labelColor);
			if (color != null && !color.equals(labelColor)) {
				ResultsCommandManager.setLabelColor(color);
//				labelColor = color;
//				updateColorButton();
			}
			
		} else if (e.getSource() == closeButton) {
			setVisible(false);
		}
	}
	
	public void stateChanged(ChangeEvent e) {
		if (e.getSource() == labelTransparencySlider) {
			ResultsCommandManager.setLabelTransparency(labelTransparencySlider.getValue());
		}
	}
	
	private void updateColorButton() {
		labelColor = mRepository.getImagePropertiesModel().getLabelsColour();
		
		Graphics2D g2 = (Graphics2D) colorImage.getGraphics();
		g2.setPaint(labelColor);
		g2.fillRect(0, 0, COLOR_BUTTON_LENGTH, COLOR_BUTTON_LENGTH);
		
		repaint();
	}
	
	public void toggleUseLabels() {
		boolean isEnabled = mRepository.getImagePropertiesModel().labelsEnabled();
		
		
		if (labelButton.isSelected() != isEnabled) {
			labelButton.removeActionListener(this);
			labelButton.setSelected(isEnabled);
			labelButton.addActionListener(this);
		}
	}
	
	public void toggleUseDescriptionLabels() {
		boolean isEnabled = mRepository.getImagePropertiesModel().descriptionLabelsEnabled();
		
		
		if (descriptionLabelButton.isSelected() != isEnabled) {
			descriptionLabelButton.removeActionListener(this);
			descriptionLabelButton.setSelected(isEnabled);
			descriptionLabelButton.addActionListener(this);
		}
	}
	
	///////////////////////////////////////////////////////////////////////////
	// Event handling
	///////////////////////////////////////////////////////////////////////////
	@Override
	public void notify(Event e) {}

	///////////////////////////////////////////////////////////////////////////
	// Brain image property event handlers
	///////////////////////////////////////////////////////////////////////////
	@Override
	public void notify(BackgroundImageEvent e) {}

	@Override
	public void notify(CrosshairColourEvent e) {}

	@Override
	public void notify(CrosshairTransparencyEvent e) {}

	@Override
	public void notify(LabelsColourEvent e) {
		updateColorButton();
	}

	@Override
	public void notify(LabelsTransparencyEvent e) {}

	@Override
	public void notify(UseCrosshairEvent e) {}

	@Override
	public void notify(UseDescriptionLabelsEvent e) {
		toggleUseDescriptionLabels();
	}

	@Override
	public void notify(UseLabelsEvent e) {
		toggleUseLabels();
	}
}