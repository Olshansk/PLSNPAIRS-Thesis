package pls.chrome.result.view;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Set;

import javax.imageio.ImageIO;
import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JScrollPane;
import javax.swing.SpringLayout;
import javax.swing.filechooser.FileNameExtensionFilter;

import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.FastScatterPlot;

import pls.chrome.result.DomainAndRangeSetter;
import pls.chrome.result.model.BrainData;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.ResultModel;
import pls.shared.GlobalVariablesFunctions;
import pls.shared.MLFuncs;

@SuppressWarnings("serial")
public class ScatterPlot extends AttachDetachOption {

	private static final int INTERSECTION = 0;
	private static final int UNION = 1;
	
	private GeneralRepository mRepository;
	
//	private String[] mResultFilePaths;
	private String mVariableType;
	private String mAbbrVariableType;

	private JLabel xAxisLabel = new JLabel("X: ");
	private BrainData[] xAxisBrainData;
	private ArrayList<Integer> xAxisCoords;
	private JComboBox xAxisFile = new JComboBox();
	private JComboBox xAxisDataType = new JComboBox();
	private JComboBox xAxisLv = new JComboBox();
	private JComboBox xAxisLag = new JComboBox();
	
	private int currentXFile = -1;
	private int currentXDataType = -1;
	private int currentXLv = -1;
	private int currentXLag = -1;

	private JLabel yAxisLabel = new JLabel("Y: ");
	private BrainData[] yAxisBrainData;
	private ArrayList<Integer> yAxisCoords;
	private JComboBox yAxisFile = new JComboBox();
	private JComboBox yAxisDataType = new JComboBox();
	private JComboBox yAxisLv = new JComboBox();
	private JComboBox yAxisLag = new JComboBox();
	
	private int currentYFile = -1;
	private int currentYDataType = -1;
	private int currentYLv = -1;
	private int currentYLag = -1;
	
	private JButton plotButton = new JButton("PLOT");
	private JRadioButton showIntersectionButton = new JRadioButton("Show common voxels", true);
	private JRadioButton showUnionButton = new JRadioButton("Show all voxels", false);
	private int currentView = -1;

	private JLabel commonVoxelLabel = new JLabel();
	private JLabel totalVoxelLabel = new JLabel();
	private JLabel correlationLabel = new JLabel();
	
	private JPanel mSelectPanel;
	private ChartPanel mScatterPlotPanel;
	private DomainAndRangeSetter mDnrSetter;
	
	private String saveFilePath = ".";
	
	public ScatterPlot(String title, GeneralRepository repository) {
		super(repository, title);
		
		mRepository = repository;
		
		setupWidgets();
		initialize();
	}
	
	private void setupWidgets() {
		mSelectPanel = new JPanel();
		SpringLayout s = new SpringLayout();
		mSelectPanel.setLayout(s);
		
		mSelectPanel.add(xAxisLabel);
		mSelectPanel.add(xAxisFile);
		mSelectPanel.add(xAxisDataType);
		mSelectPanel.add(xAxisLv);
		mSelectPanel.add(xAxisLag);
		mSelectPanel.add(yAxisLabel);
		mSelectPanel.add(yAxisFile);
		mSelectPanel.add(yAxisDataType);
		mSelectPanel.add(yAxisLv);
		mSelectPanel.add(yAxisLag);
		mSelectPanel.add(plotButton);
		
		s.putConstraint(SpringLayout.NORTH, xAxisLabel, 12, SpringLayout.NORTH, mSelectPanel);
		s.putConstraint(SpringLayout.WEST, xAxisLabel, 0, SpringLayout.WEST, mSelectPanel);
		s.putConstraint(SpringLayout.NORTH, xAxisFile, 10, SpringLayout.NORTH, mSelectPanel);
		s.putConstraint(SpringLayout.WEST, xAxisFile, 0, SpringLayout.EAST, xAxisLabel);
		s.putConstraint(SpringLayout.NORTH, xAxisDataType, 10, SpringLayout.NORTH, mSelectPanel);
		s.putConstraint(SpringLayout.WEST, xAxisDataType, 10, SpringLayout.EAST, xAxisFile);
		s.putConstraint(SpringLayout.NORTH, xAxisLv, 10, SpringLayout.NORTH, mSelectPanel);
		s.putConstraint(SpringLayout.WEST, xAxisLv, 10, SpringLayout.EAST, xAxisDataType);
		s.putConstraint(SpringLayout.NORTH, xAxisLag, 10, SpringLayout.NORTH, mSelectPanel);
		s.putConstraint(SpringLayout.WEST, xAxisLag, 10, SpringLayout.EAST, xAxisLv);

		s.putConstraint(SpringLayout.NORTH, yAxisLabel, 12, SpringLayout.SOUTH, xAxisLabel);
		s.putConstraint(SpringLayout.WEST, xAxisLabel, 0, SpringLayout.WEST, mSelectPanel);
		s.putConstraint(SpringLayout.NORTH, yAxisFile, 5, SpringLayout.SOUTH, xAxisFile);
		s.putConstraint(SpringLayout.WEST, yAxisFile, 0, SpringLayout.EAST, yAxisLabel);
		s.putConstraint(SpringLayout.NORTH, yAxisDataType, 5, SpringLayout.SOUTH, xAxisDataType);
		s.putConstraint(SpringLayout.WEST, yAxisDataType, 10, SpringLayout.EAST, yAxisFile);
		s.putConstraint(SpringLayout.NORTH, yAxisLv, 5, SpringLayout.SOUTH, xAxisLv);
		s.putConstraint(SpringLayout.WEST, yAxisLv, 10, SpringLayout.EAST, yAxisDataType);
		s.putConstraint(SpringLayout.NORTH, yAxisLag, 5, SpringLayout.SOUTH, xAxisLag);
		s.putConstraint(SpringLayout.WEST, yAxisLag, 10, SpringLayout.EAST, yAxisLv);
		s.putConstraint(SpringLayout.SOUTH, mSelectPanel, 10, SpringLayout.SOUTH, yAxisFile);
		
		s.putConstraint(SpringLayout.NORTH, plotButton, (xAxisDataType.getPreferredSize().height / 2) + 10, SpringLayout.NORTH, mSelectPanel);
		s.putConstraint(SpringLayout.EAST, plotButton, 0, SpringLayout.EAST, mSelectPanel);
		
		JPanel selectPanel = new JPanel();
		selectPanel.setAlignmentX(JPanel.CENTER_ALIGNMENT);
		selectPanel.add(mSelectPanel);
		
		mDnrSetter = new DomainAndRangeSetter(1, false);
		selectPanel.add(mDnrSetter);
		
		int width = selectPanel.getPreferredSize().width;
		int height = selectPanel.getPreferredSize().height + 20;
		JScrollPane selectScrollPane = new JScrollPane(selectPanel);
		selectScrollPane.setPreferredSize(new Dimension(width, height));
		
		mScatterPlotPanel = new ChartPanel(null);
		mScatterPlotPanel.setAlignmentX(JPanel.CENTER_ALIGNMENT);
		
		JPanel buttonPanel = new JPanel();
		buttonPanel.add(mAttachDetachButton);
		buttonPanel.add(showUnionButton);
		buttonPanel.add(totalVoxelLabel);
		buttonPanel.add(showIntersectionButton);
		buttonPanel.add(commonVoxelLabel);
		buttonPanel.add(correlationLabel);
		
		ButtonGroup buttonGroup = new ButtonGroup();
		buttonGroup.add(showIntersectionButton);
		buttonGroup.add(showUnionButton);
		
		setLayout(new BorderLayout());
		add(selectScrollPane, BorderLayout.NORTH);
		add(mScatterPlotPanel, BorderLayout.CENTER);
		add(buttonPanel, BorderLayout.SOUTH);
		
		plotButton.addActionListener(this);
		showUnionButton.addActionListener(this);
		showIntersectionButton.addActionListener(this);
	}
	
	public void initialize() {
		mResultFilePaths = new ArrayList<String>();
		xAxisFile.removeActionListener(this);
		yAxisFile.removeActionListener(this);
		xAxisFile.removeAllItems();
		yAxisFile.removeAllItems();

		Set<String> resultFiles = mRepository.getModels();
		for (String filePath : resultFiles) {
			mResultFilePaths.add(filePath);
			
			String fileName = filePath.substring(filePath.lastIndexOf(File.separator) + 1, filePath.length());
			xAxisFile.addItem(fileName);
			yAxisFile.addItem(fileName);
		}
		
		// Uses the first result file as the default result file to select
		// for both axes.
		ResultModel model = mRepository.getGeneral(mResultFilePaths.get(0));
		updateComboBoxes("x", model);
		updateComboBoxes("y", model);
		
		Dimension dim = new Dimension(250, xAxisFile.getPreferredSize().height);
		xAxisFile.setPreferredSize(dim);
		yAxisFile.setPreferredSize(dim);
		
		dim = new Dimension(150, xAxisDataType.getPreferredSize().height);
		xAxisDataType.setPreferredSize(dim);
		yAxisDataType.setPreferredSize(dim);
		
		int width = xAxisLabel.getPreferredSize().width
		  + xAxisFile.getPreferredSize().width
		  + xAxisDataType.getPreferredSize().width
		  + xAxisLv.getPreferredSize().width
		  + xAxisLag.getPreferredSize().width
		  + plotButton.getPreferredSize().width
		  + 40;

		int height = (2 * xAxisDataType.getPreferredSize().height) + 25;
		mSelectPanel.setPreferredSize(new Dimension(width, height));
		
		xAxisFile.addActionListener(this);
		yAxisFile.addActionListener(this);
	}

	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == mAttachDetachButton) {
			super.actionPerformed(e);
		} else if (e.getSource() == plotButton) {
			plotButtonAction();
			mDnrSetter.plotButtonAction();
			mDnrSetter.updateInputFields();
		} else if (e.getSource() == showUnionButton
				|| e.getSource() == showIntersectionButton) {
			mDnrSetter.clearInputFields();
			plotButtonAction();
			mDnrSetter.plotButtonAction();
			mDnrSetter.updateInputFields();
		} else {
			if (e.getSource() == xAxisFile) {
				int xFile = xAxisFile.getSelectedIndex();
				if (xFile == currentXFile) {
					return;
				}
				ResultModel model = mRepository.getGeneral(mResultFilePaths.get(xFile) );
				updateComboBoxes("x", model);
			
			} else if (e.getSource() == yAxisFile) {
				int yFile = yAxisFile.getSelectedIndex();
				if (yFile == currentYFile) {
					return;
				}
				ResultModel model = mRepository.getGeneral(mResultFilePaths.get(yFile) );
				updateComboBoxes("y", model);
			}
			mDnrSetter.clearInputFields();
		}
	}
	
	private void updateComboBoxes(String axis, ResultModel model) {
		Set<String> dataTypes = model.getBrainDataTypes();
		Iterator<String> iter = dataTypes.iterator();
		mVariableType = model.getVariableType();
		mAbbrVariableType = model.getAbbrVariableType();
		
		if (axis.equals("x")) {
			xAxisBrainData = new BrainData[dataTypes.size()];
			xAxisCoords = new ArrayList<Integer>(model.getFilteredCoordinates() );
			xAxisDataType.removeActionListener(this);
			xAxisLv.removeActionListener(this);
			xAxisLag.removeActionListener(this);
			xAxisDataType.removeAllItems();
			xAxisLv.removeAllItems();
			xAxisLag.removeAllItems();
			
			int i = 0;
			while (iter.hasNext()) {
				String dataType = iter.next();
				xAxisBrainData[i] = model.getBrainData(dataType);
				xAxisDataType.addItem(dataType);
				i++;
			}
			
			int numLvs = xAxisBrainData[0].getNumLvs();
			for (int j = 0; j != numLvs; j++) {
				xAxisLv.addItem(mVariableType + " #" + (j + 1));
			}
			
			int numLags = model.getWindowSize();
			for (int j = 0; j != numLags; j++) {
				xAxisLag.addItem("Lag #" + j);
			}

			xAxisDataType.addActionListener(this);
			xAxisLv.addActionListener(this);
			xAxisLag.addActionListener(this);
		} else {
			yAxisBrainData = new BrainData[dataTypes.size()];
			yAxisCoords = new ArrayList<Integer>(model.getFilteredCoordinates() );
			yAxisDataType.removeActionListener(this);
			yAxisLv.removeActionListener(this);
			yAxisLag.removeActionListener(this);
			yAxisDataType.removeAllItems();
			yAxisLv.removeAllItems();
			yAxisLag.removeAllItems();
			
			int i = 0;
			while (iter.hasNext()) {
				String dataType = iter.next();
				yAxisBrainData[i] = model.getBrainData(dataType);
				yAxisDataType.addItem(dataType);
				i++;
			}
			
			int numLvs = yAxisBrainData[0].getNumLvs();
			for (int j = 0; j != numLvs; j++) {
				yAxisLv.addItem(mVariableType + " #" + (j + 1));
			}
			
			int numLags = model.getWindowSize();
			for (int j = 0; j != numLags; j++) {
				yAxisLag.addItem("Lag #" + j);
			}

			yAxisDataType.addActionListener(this);
			yAxisLv.addActionListener(this);
			yAxisLag.addActionListener(this);
		}
	}
	
	private void plotButtonAction() {
		int xFile = xAxisFile.getSelectedIndex();
		int xDataType = xAxisDataType.getSelectedIndex();
		int xLv = xAxisLv.getSelectedIndex();
		int xLag = xAxisLag.getSelectedIndex();

		int yFile = yAxisFile.getSelectedIndex();
		int yDataType = yAxisDataType.getSelectedIndex();
		int yLv = yAxisLv.getSelectedIndex();
		int yLag = yAxisLag.getSelectedIndex();
		
		int view;
		if (showUnionButton.isSelected()) {
			view = UNION;
		} else {
			view = INTERSECTION;
		}
		
		if(xDataType == currentXDataType && yDataType == currentYDataType
				&& xLv == currentXLv && yLv == currentYLv
				&& xLag == currentXLag && yLag == currentYLag
				&& xFile == currentXFile && yFile == currentYFile
				&& view == currentView
				&& mDnrSetter.noFieldsEmpty()) {
			return;
		} else {
			currentXDataType = xDataType;
			currentXLv = xLv;
			currentXLag = xLag;
			currentXFile = xFile;
			currentYDataType = yDataType;
			currentYLv = yLv;
			currentYLag = yLag;
			currentYFile = yFile;
			currentView = view;
		}
		
		BrainData xBrainData = xAxisBrainData[xDataType];
		BrainData yBrainData = yAxisBrainData[yDataType];
		
		
		ArrayList<Integer> unionCoords = new ArrayList<Integer>(xAxisCoords);
		unionCoords.addAll(yAxisCoords);
//		ArrayList<Integer> unionCoords = MLFuncs.unionSortedArrays(xAxisCoords, yAxisCoords);
		
		ArrayList<Integer> intersectionCoords = new ArrayList<Integer>(yAxisCoords);
		intersectionCoords.retainAll(yAxisCoords);
//		ArrayList<Integer> intersectionCoords = MLFuncs.intersectionSortedArrays(xAxisCoords, yAxisCoords);
		
		totalVoxelLabel.setText("(" + unionCoords.size() + ")");
		commonVoxelLabel.setText("(" + intersectionCoords.size() + ", " + (100.0 * intersectionCoords.size() / unionCoords.size()) + "%)");
		
		ArrayList<Integer> coords;
		if (showUnionButton.isSelected()) {
			coords = unionCoords;
		} else {
			coords = intersectionCoords;
		}
		
		float[][] plotValues = new float[2][coords.size()];
		for (int i = 0; i != coords.size(); i++) {
			plotValues[0][i] = (float) xBrainData.getValue1D(coords.get(i) - 1, xLag, xLv);
			plotValues[1][i] = (float) yBrainData.getValue1D(coords.get(i) - 1, yLag, yLv);
		}
		
		correlationLabel.setText("Correlation = " + getCorrelationValue(plotValues));
		
		String xFileName = (String) xAxisFile.getItemAt(xFile);
		String yFileName = (String) yAxisFile.getItemAt(yFile);
		if (xFileName.length() > 20) {
			xFileName = xFileName.substring(0, 20) + "...";
		}
		if (yFileName.length() > 20) {
			yFileName = yFileName.substring(0, 20) + "...";
		}
		
		ValueAxis xAxis = new NumberAxis(xFileName + ", " + xAxisDataType.getItemAt(xDataType) + ", " + mAbbrVariableType + " #" + (xLv + 1) + ", Lag #" + xLag);
		ValueAxis yAxis = new NumberAxis(yFileName + ", " + yAxisDataType.getItemAt(yDataType) + ", " + mAbbrVariableType + " #" + (yLv + 1) + ", Lag #" + yLag);
		FastScatterPlot plot = new FastScatterPlot(plotValues, xAxis, yAxis);
		
		mDnrSetter.setAxes(xAxis, yAxis, 0);
		JFreeChart chart = new JFreeChart(plot);
		mScatterPlotPanel.setChart(chart);
	}
	
	private double getCorrelationValue(float[][] values) {
		double result = 0;
		float xMean = MLFuncs.avg(values[0]);
		double xStdev = MLFuncs.stdev(xMean, values[0], false);
		float yMean = MLFuncs.avg(values[1]);
		double yStdev = MLFuncs.stdev(yMean, values[1], false);
		double numValues = values[0].length;
		
		for (int i = 0; i != values[0].length; i++) {
			float xValue = values[0][i];
			float yValue = values[1][i];
			
			if (xValue == 0 || yValue == 0) {
				numValues--;
				continue;
			}
			
			xValue -= xMean;
			yValue -= yMean;
			result += (xValue * yValue);
		}
		
		result /= ((numValues - 1) * xStdev * yStdev);
		
		return result;
	}
	
	public void doSaveAs() {
		JFreeChart chart = mScatterPlotPanel.getChart();
		if (chart != null) {
				
			// Retrieves the different image formats the plots can be saved as.
			String[] formatNames = ImageIO.getWriterFormatNames();
			ArrayList<String> filteredNames = new ArrayList<String>();
			for (String s : formatNames) {
				String uppercase = s.toUpperCase();
				if (!filteredNames.contains(uppercase)) {
					filteredNames.add(uppercase);
				}
			}
				
			JFileChooser chooser = new JFileChooser(saveFilePath);
			for (String s : filteredNames) {
				chooser.addChoosableFileFilter(new FileNameExtensionFilter(s, s));
			}
				
			int option = chooser.showDialog(this, "Save Plot As ...");
			if (option == JFileChooser.APPROVE_OPTION) {
				File file = chooser.getSelectedFile();
				
				saveFilePath = file.getParent();
				String fileName = file.getAbsolutePath();
				
				// Sets the selected file extension based on the selected file filter.
				String extension = chooser.getFileFilter().getDescription();
				if (!fileName.endsWith("." + extension) ) {
					fileName += "." + extension;
					file = new File(fileName);
				}
				
				try {
					BufferedImage image = chart.createBufferedImage(mScatterPlotPanel.getWidth(), mScatterPlotPanel.getHeight());
					ImageIO.write(image, extension, file);
				} catch (Exception e) {
					GlobalVariablesFunctions.showErrorMessage("Unable to save the current plot.");
				}
			}
		} else {
			GlobalVariablesFunctions.showErrorMessage("No plot has been loaded yet.");
		}
	}
}