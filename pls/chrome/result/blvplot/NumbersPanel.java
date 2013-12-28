package pls.chrome.result.blvplot;

import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.GridLayout;

import javax.swing.BoxLayout;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SpringLayout;
import javax.swing.border.LineBorder;

import pls.shared.PValue;

import pls.chrome.result.model.BrainData;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.ResultModel;

@SuppressWarnings("serial")
public class NumbersPanel extends JPanel {
	private JLabel brainValLabel = new JLabel();
	private JLabel pValLabel = new JLabel();
	private JLabel dimensionsLabel = null;
	private ColorBarPanel colorBarPanel = null;

	GeneralRepository mRepository = null;
	
	public NumbersPanel (GeneralRepository repository) {
		mRepository = repository;
		layOutWidgets();
	}
	
	private void layOutWidgets() {
		setLayout(new BoxLayout(this, BoxLayout.X_AXIS));
		
		JPanel innerPanel = new JPanel();
		innerPanel.setLayout(new GridLayout(3, 1) );
		
		setAlignmentX(JPanel.LEFT_ALIGNMENT);
		
		ResultModel model = mRepository.getGeneral();
		BrainData bData = model.getBrainData();
		
		dimensionsLabel = new JLabel("\tBrain Dimensions:  XYZ  =  " 
				+ bData.getWidth(BrainData.AXIAL) + "  x  "
				+ bData.getHeight(BrainData.AXIAL) + "  x  "
				+ bData.getNumSlices(BrainData.AXIAL) );
		
		colorBarPanel = new ColorBarPanel(mRepository);
		
		innerPanel.add(brainValLabel);
		innerPanel.add(dimensionsLabel);
		innerPanel.add(pValLabel);
		add(innerPanel);
		add(colorBarPanel);
	}
	
	public void updateBrainValue(double brainVal) {
		brainValLabel.setText("\tBrain Value: " + Double.toString(brainVal));
		
		invalidate();
	}
	
	public void clearValues() {
		brainValLabel.setText("");
	}
	
	public void updatePValue() {
		double threshold = 0.0;
		
		// If we're using a global colour scale, calculate pvalue based on that
		if (mRepository.getUseGlobalScale() ) {
			threshold = mRepository.getGlobalColourScale()[2];
		}
		// Otherwise calculate pvalue from the currently selected lv
		else {
			threshold = mRepository.getGeneral().getBrainData().getColourScaleModel().getColourScale()[2];
		}
		
		double pValue = PValue.ThresholdToPValue(threshold, 0, 1);
		pValLabel.setText("PValue: " + pValue);
	}

	private void updateDimensions(){
		ResultModel model = mRepository.getGeneral();
		BrainData bData = model.getBrainData();

		dimensionsLabel.setText("\tBrain Dimensions:  XYZ  =  "
				+ bData.getWidth(BrainData.AXIAL) + "  x  "
				+ bData.getHeight(BrainData.AXIAL) + "  x  "
				+ bData.getNumSlices(BrainData.AXIAL));
	}
	public void refreshPanel() {
		updatePValue();
		updateDimensions();
		colorBarPanel.refreshPanel();
	}
}
