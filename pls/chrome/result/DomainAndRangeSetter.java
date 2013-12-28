package pls.chrome.result;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.text.DecimalFormat;

import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTextField;

import org.jfree.chart.axis.ValueAxis;

@SuppressWarnings("serial")
public class DomainAndRangeSetter extends JPanel implements ActionListener, KeyListener {
	
	public static final int ENTER_KEY = 10;
	
	private JTextField mRangeMin = null;
	private JTextField mRangeMax = null;
	private JTextField mDomainMin = null;
	private JTextField mDomainMax = null;
	private JButton mPlotButton = null;
	private boolean includePlotButton;
	
	ValueAxis mRangeAxis = null;
	ValueAxis mDomainAxis = null;
	
	Double[][] mRange = null;
	Double[][] mDomain = null;
	
	int mPlotIndex = -1;
	
	/**
	 * Because the DomainAndRangeSetter remembers the domain and range of
	 * several plots, the number of plots to save must be provided.
	 */
	public DomainAndRangeSetter(int numPlots, boolean includePlotButton) {
		this.includePlotButton = includePlotButton;
		setupWidgets();
		
		mRange = new Double[numPlots][2];
		mDomain = new Double[numPlots][2];
	}
	
	public DomainAndRangeSetter(int numPlots) {
		this(numPlots, true);
	}
	
	public void setNumPlots(int numPlots) {
		mRange = new Double[numPlots][2];
		mDomain = new Double[numPlots][2];
	}
	
	private void setupWidgets() {
		JPanel rangePanel = new JPanel();
		rangePanel.add(new JLabel("Range:" ));
		mRangeMin = new JTextField(8);
		rangePanel.add(mRangeMin);
		rangePanel.add(new JLabel("-"));
		mRangeMax = new JTextField(8);
		rangePanel.add(mRangeMax);
		
		JPanel domainPanel = new JPanel();
		domainPanel.add(new JLabel("Domain:" ));
		mDomainMin = new JTextField(8);
		domainPanel.add(mDomainMin);
		domainPanel.add(new JLabel("-"));
		mDomainMax = new JTextField(8);
		domainPanel.add(mDomainMax);
		
		setLayout(new BoxLayout(this, BoxLayout.Y_AXIS) );
		add(domainPanel);
		add(rangePanel);
		mPlotButton = new JButton("PLOT");
		if (includePlotButton) {
			add(mPlotButton);
		}
		
		mPlotButton.addActionListener(this);
		mRangeMin.addKeyListener(this);
		mRangeMax.addKeyListener(this);
		mDomainMin.addKeyListener(this);
		mDomainMax.addKeyListener(this);
	}
	
	/**
	 * Called by a plot tab when it has updated its plot (perhaps due to
	 * changing LV, CV, or whatever).
         * @param plotIndex This is the LV,CV for which we want to set the axes.
	 */
	public void setAxes(ValueAxis domain, ValueAxis range, int plotIndex) {
                mDomainAxis = domain;
		mRangeAxis = range;
		mPlotIndex = plotIndex;
	}
	
	public void updateInputFields() {
		String lower;
		String upper;
		if (mRange[mPlotIndex][0] == null) {
			lower = truncate(mRangeAxis.getRange().getLowerBound());
			upper = truncate(mRangeAxis.getRange().getUpperBound());
			
			mRangeMin.setText(lower);
			mRangeMax.setText(upper);
		}
		else {
			mRangeAxis.setRange(mRange[mPlotIndex][0], mRange[mPlotIndex][1]);
			lower = truncate(mRange[mPlotIndex][0]);
			upper = truncate(mRange[mPlotIndex][1]);
			
			mRangeMin.setText(lower);
			mRangeMax.setText(upper);
		}
		
		if (mDomain[mPlotIndex][0] == null) {
			lower = truncate(mDomainAxis.getRange().getLowerBound());
			upper = truncate(mDomainAxis.getRange().getUpperBound());
			
			mDomainMin.setText(lower);
			mDomainMax.setText(upper);
		}
		else {
			mDomainAxis.setRange(mDomain[mPlotIndex][0], mDomain[mPlotIndex][1]);
			lower = truncate(mDomain[mPlotIndex][0]);
			upper = truncate(mDomain[mPlotIndex][1]);
			
			mDomainMin.setText(lower);
			mDomainMax.setText(upper);
		}
	}
	
	private String truncate(double value) {
		DecimalFormat formatter = new DecimalFormat("#####.###");	
		return formatter.format(value);
	}

	/**
	 * Called when the user clicks on the PLOT button
	 */
	public void actionPerformed(ActionEvent e) {
		plotButtonAction();
	}
    
	public void keyPressed(KeyEvent e) {
		
		// We only handle the case where the enter key is pressed.
		if (e.getKeyChar() != ENTER_KEY) {
			return;
		}
		
		Object source = e.getSource();
		if (source == mRangeMin || source == mRangeMax
			|| source == mDomainMin || source == mDomainMax) {
			
			plotButtonAction();
    	}
	}
	
	public void plotButtonAction() {
		String rangeMin = mRangeMin.getText().trim();
		String rangeMax = mRangeMax.getText().trim();
		String domainMin = mDomainMin.getText().trim();
		String domainMax = mDomainMax.getText().trim();
		
		// If any of the input fields are blank or contain invalid values,
		// then the currently-set values are used instead.
		try {
			mRange[mPlotIndex][0] = Double.parseDouble(rangeMin);
		} catch (NumberFormatException e) {
			mRange[mPlotIndex][0] = mRangeAxis.getRange().getLowerBound();
		}
		try {
			mRange[mPlotIndex][1] = Double.parseDouble(rangeMax);
		} catch (NumberFormatException e) {
			mRange[mPlotIndex][1] = mRangeAxis.getRange().getUpperBound();
		}
		try {
			mDomain[mPlotIndex][0] = Double.parseDouble(domainMin);
		} catch (NumberFormatException e) {
			mDomain[mPlotIndex][0] = mDomainAxis.getRange().getLowerBound();
		}
		try {
			mDomain[mPlotIndex][1] = Double.parseDouble(domainMax);
		} catch (NumberFormatException e) {
			mDomain[mPlotIndex][1] = mDomainAxis.getRange().getUpperBound();
		}
			
		if (mRange[mPlotIndex][0] > mRange[mPlotIndex][1]) {
			JOptionPane.showMessageDialog(null, "The lower bound can not be greater than the upper bound for the range.", "Error", JOptionPane.ERROR_MESSAGE);
			mRange[mPlotIndex][0] = mRangeAxis.getRange().getLowerBound();
			mRange[mPlotIndex][1] = mRangeAxis.getRange().getUpperBound();
		}
		
		if (mDomain[mPlotIndex][0] > mDomain[mPlotIndex][1]) {
			JOptionPane.showMessageDialog(null, "The lower bound can not be greater than the upper bound for the domain.", "Error", JOptionPane.ERROR_MESSAGE);
			mDomain[mPlotIndex][0] = mDomainAxis.getRange().getLowerBound();
			mDomain[mPlotIndex][1] = mDomainAxis.getRange().getUpperBound();
		}
		
		mRangeAxis.setRange(mRange[mPlotIndex][0], mRange[mPlotIndex][1]);
		mDomainAxis.setRange(mDomain[mPlotIndex][0], mDomain[mPlotIndex][1]);
	}
	
	public void clearInputFields() {
		mRangeMin.setText("");
		mRangeMax.setText("");
		mDomainMin.setText("");
		mDomainMax.setText("");
	}
	
	public boolean noFieldsEmpty() {
		return !mRangeMin.getText().equals("") 
			&& !mRangeMax.getText().equals("") 
			&& !mDomainMin.getText().equals("") 
			&& !mDomainMax.getText().equals("");
	}
	
	public void keyTyped(KeyEvent e) {}
	public void keyReleased(KeyEvent e) {}
}
