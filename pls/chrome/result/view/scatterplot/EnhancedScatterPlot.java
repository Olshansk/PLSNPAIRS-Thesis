package pls.chrome.result.view.scatterplot;
 
import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Set;
import java.util.TreeSet;
 
import javax.imageio.ImageIO;
import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JComponent;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JScrollPane;
import javax.swing.SpringLayout;
import javax.swing.filechooser.FileNameExtensionFilter;
 
import org.jfree.chart.JFreeChart;
import pls.chrome.result.blvplot.PlotTypeTabbedPane;
 
import pls.chrome.result.controller.observer.Event;
import pls.chrome.result.controller.observer.datachange.DataChangeObserver;
import pls.chrome.result.controller.observer.datachange.FlipVolumeEvent;
import pls.chrome.result.controller.observer.datachange.InvertedLvEvent;
import pls.chrome.result.controller.observer.datachange.LoadedVolumesEvent;
import pls.chrome.result.controller.observer.filters.BrainFilterEvent;
import pls.chrome.result.controller.observer.filters.FiltersObserver;
import pls.chrome.result.controller.observer.filters.IncorrectLagsSelectedEvent;
import pls.chrome.result.controller.observer.filters.SliceFiltersEvent;
import pls.chrome.result.controller.observer.filters.ViewedLvsEvent;
import pls.chrome.result.model.BrainData;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.ResultModel;
import pls.chrome.result.view.AttachDetachOption;
import pls.chrome.shared.ImageWriter;
import pls.shared.GlobalVariablesFunctions;
import pls.shared.MLFuncs;
 
@SuppressWarnings("serial")
public class EnhancedScatterPlot extends AttachDetachOption 
		implements DataChangeObserver, FiltersObserver{
 
    //private static final int INTERSECTION = 0;
    //private static final int UNION = 1;
     
    private GeneralRepository mRepository;
    private PlotTypeTabbedPane plotTabbedPane;
    
    private boolean plotReady = false;

    private String mVariableType;
    private String mAbbrVariableType;
 
    private JLabel xAxisLabel = new JLabel("X: ");
    private BrainData[] xAxisBrainData;
    private TreeSet<Integer> xAxisCoords; //Filtered xAxis model coords.
    private JComboBox xAxisFile = new JComboBox();
    private JComboBox xAxisDataType = new JComboBox();
    private JComboBox xAxisLv = new JComboBox();
    private JComboBox xAxisLag = new JComboBox();
     
    private int currentXFile = -1;
 
    private JLabel yAxisLabel = new JLabel("Y: ");
    private BrainData[] yAxisBrainData;
    private TreeSet<Integer> yAxisCoords; //Filtered yAxis model coords.
    private JComboBox yAxisFile = new JComboBox();
    private JComboBox yAxisDataType = new JComboBox();
    private JComboBox yAxisLv = new JComboBox();
    private JComboBox yAxisLag = new JComboBox();
     
    private int currentYFile = -1;
     
    private JButton plotButton = new JButton("PLOT");
	private JButton maskButton;
	private JButton cancelButton;
 
    private JLabel commonVoxelLabel = new JLabel();
    private JLabel totalVoxelLabel = new JLabel();
    private JLabel correlationLabel = new JLabel();
     
    private JPanel mSelectPanel;
    private ScatterPlotCanvas mScatterPlotPanel;
        
     
    public EnhancedScatterPlot(String title, GeneralRepository repository,
    										PlotTypeTabbedPane tabbedPane) {
        super(repository, title);
        
        plotTabbedPane = tabbedPane;
        mRepository = repository;
        mRepository.getPublisher().registerObserver(this);
         
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
         
        s.putConstraint(SpringLayout.NORTH, plotButton, 
				(xAxisDataType.getPreferredSize().height / 2) + 10,
				SpringLayout.NORTH, mSelectPanel);
        s.putConstraint(SpringLayout.EAST, plotButton, 0, SpringLayout.EAST, mSelectPanel);
         
        JPanel selectPanel = new JPanel();
        selectPanel.setAlignmentX(JPanel.CENTER_ALIGNMENT);
        selectPanel.add(mSelectPanel);
         
        int width = selectPanel.getPreferredSize().width;
        int height = selectPanel.getPreferredSize().height + 20;
        JScrollPane selectScrollPane = new JScrollPane(selectPanel);
        selectScrollPane.setPreferredSize(new Dimension(width, height));
         
		mScatterPlotPanel = new ScatterPlotCanvas(mRepository,this);
        mScatterPlotPanel.setAlignmentX(JComponent.CENTER_ALIGNMENT);
         
        maskButton = new JButton("Apply mask");
		maskButton.setEnabled(false);
        maskButton.setActionCommand("apply mask");
        maskButton.addActionListener(this);

		JPanel maskControl = new JPanel(new BorderLayout());
		cancelButton = new JButton("Cancel Mask");
		cancelButton.setEnabled(false);
		maskControl.add(cancelButton, BorderLayout.NORTH);
		maskControl.add(maskButton, BorderLayout.SOUTH);

		JPanel statisticalControl = new JPanel(new BorderLayout());
		statisticalControl.add(totalVoxelLabel, BorderLayout.NORTH);
		statisticalControl.add(commonVoxelLabel, BorderLayout.SOUTH);

		JPanel buttonPanel = new JPanel();
        buttonPanel.add(maskControl);
		buttonPanel.add(mAttachDetachButton);
		buttonPanel.add(statisticalControl);
        buttonPanel.add(correlationLabel);
         
        setLayout(new BorderLayout());
        add(selectScrollPane, BorderLayout.NORTH);
        add(mScatterPlotPanel, BorderLayout.CENTER);
        add(buttonPanel, BorderLayout.SOUTH);

		cancelButton.addActionListener(this);
        plotButton.addActionListener(this);
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
    
    @Override
    /**
     * Action that occurs when the user decides to close the window
     */
	public void windowClosing(WindowEvent e) {
		attachDetachAction();
	}
    
	@Override
    public void actionPerformed(ActionEvent e) {
        if (e.getActionCommand().equals("apply mask") ) {
            mScatterPlotPanel.createMask();
		}
		else if (e.getSource() == mAttachDetachButton) {
			attachDetachAction();
		} else if (e.getSource() == plotButton) {
        	
        	if(!plotReady){//only do this once in the entire 
        				//execution of the result viewer. 
        		plotReady = true;
        	
	        	//fire a selection event so that the "save scatterplot as ..."
	        	//menu item is enabled. See lines 217-232 in ResultMenuBar2.java
        		Component selected = plotTabbedPane.getSelectedComponent();
        		
        		//if we select a component again after it has already been 
        		//selected no ChangeEvent will be generated so we need to
        		//select a different tab and then flip back.
        		plotTabbedPane.setSelectedIndex(0);
        		plotTabbedPane.setSelectedComponent(selected);
	        	
        	}
        	
            plotButtonAction();
		} else if(e.getSource() == cancelButton){
			if(!mScatterPlotPanel.getCreateNewPolygon()){
				cancelButton.setEnabled(false);
				mScatterPlotPanel.destroyCurrentPolygon();
				mScatterPlotPanel.repaint();
			}
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
        }
    }
    
	private void attachDetachAction(){
		//System.out.println("Currently " + mScatterPlotPanel.getDimensions());
//		mScatterPlotPanel.cacheSize(
//				mScatterPlotPanel.getDimensions()
//		);
				
		if (mAttachDetachButton.getText().equals("Detach")) {
			this.detachWindow();
		} 
		else {
			mAttachDetachButton.setText("Detach");
			mDetachedFrame.remove(this);
			mDetachedFrame.setVisible(false);
			
			PlotTypeTabbedPane tabbedPane = plotTabbedPane;
			Component Cluster = tabbedPane.getComponent(2);
	
			tabbedPane.remove(Cluster);
			tabbedPane.add("Scatter Plot", this);
			tabbedPane.add("Cluster Report", Cluster);
			tabbedPane.setSelectedComponent(this);
		}
		/*System.out.println("Too soon: " + mScatterPlotPanel.getDimensions());
		mScatterPlotPanel.translatePolygon(
				mScatterPlotPanel.getDimensions()
		);*/	

	}
	
    private void updateComboBoxes(String axis, ResultModel model) {
        Set<String> dataTypes = model.getBrainDataTypes();
        Iterator<String> iter = dataTypes.iterator();
        mVariableType = model.getVariableType();
        mAbbrVariableType = model.getAbbrVariableType();
         
        if (axis.equals("x")) {
            xAxisBrainData = new BrainData[dataTypes.size()];
            xAxisCoords = model.getFilteredCoordinates();
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
            yAxisCoords = model.getFilteredCoordinates();
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
        mScatterPlotPanel.destroyCurrentPolygon();
        int xFile = xAxisFile.getSelectedIndex();
        int xDataType = xAxisDataType.getSelectedIndex();
        int xLv = xAxisLv.getSelectedIndex();
        int xLag = xAxisLag.getSelectedIndex();
 
        int yFile = yAxisFile.getSelectedIndex();
        int yDataType = yAxisDataType.getSelectedIndex();
        int yLv = yAxisLv.getSelectedIndex();
        int yLag = yAxisLag.getSelectedIndex();

		double selectedCordsPercent;
		double corrVal;
		DecimalFormat df = new DecimalFormat("#.######");

		BrainData xBrainData = xAxisBrainData[xDataType];
        BrainData yBrainData = yAxisBrainData[yDataType];

		ResultModel xmodel = mRepository.getGeneral(mResultFilePaths.get(xFile));
		ResultModel ymodel = mRepository.getGeneral(mResultFilePaths.get(yFile));

		xAxisCoords = xmodel.getFilteredCoordinates();
		yAxisCoords = ymodel.getFilteredCoordinates();
		
		TreeSet<Integer> totalCommonCoords = new TreeSet<Integer>(
				xmodel.getCoordinatesSet());
		TreeSet<Integer> intersectionCoords = new TreeSet<Integer>(xAxisCoords);

  
		currentXFile = xFile;
		currentYFile = yFile;

		totalCommonCoords.retainAll(ymodel.getCoordinatesSet());
        intersectionCoords.retainAll(yAxisCoords);

		selectedCordsPercent = 100.0 * intersectionCoords.size()
				/ totalCommonCoords.size();
        commonVoxelLabel.setText("Selected voxels: " + intersectionCoords.size());
		totalVoxelLabel.setText("Percentage of common voxels: " +
				df.format(selectedCordsPercent) + "% ");
        TreeSet<Integer> coords = intersectionCoords;

		//exception is thrown here if a value dne for a given cord.
        float[][] plotValues = new float[2][coords.size()];
        int i = 0;
        Iterator<Integer> iter = coords.iterator(); 
        while (iter.hasNext() ) {
            int coord = iter.next();
			plotValues[0][i] = (float) xBrainData.getValue1DCorr(coord, xLag, xLv);
			plotValues[1][i] = (float) yBrainData.getValue1DCorr(coord, yLag, yLv);
            ++i;
        }


		corrVal = getCorrelationValue(plotValues);
		if(!Double.isNaN(corrVal)){
			correlationLabel.setText("Correlation = " + df.format(
					corrVal));
		}else{
			correlationLabel.setText("Correlation = " + corrVal);
		}
		
        //correlationLabel.setText("Correlation = " + getCorrelationValue(plotValues));
		//Look up name from index {x,y}DataType.
        String xDataTypeString = (String)xAxisDataType.getItemAt(xDataType);
        String yDataTypeString = (String)yAxisDataType.getItemAt(yDataType);
 
        mScatterPlotPanel.setInfo(
				mRepository.getGeneral(mResultFilePaths.get(xFile)),
				mRepository.getGeneral(mResultFilePaths.get(yFile)),
                xDataTypeString, yDataTypeString, xLv, yLv, xLag, yLag);
        mScatterPlotPanel.repaint();
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
             
             
            xValue -= xMean;
            yValue -= yMean;
            result += (xValue * yValue);
        }
         
        result /= ((numValues - 1) * xStdev * yStdev);
         
        return result;
    }
    /**
     * Write the scatter plot out as an image file.
     */
    public void doSaveAs(){
    	ImageWriter.saveESP(this);
    }

    /**
     * Return whether there exists a scatter plot that can be saved to an image.
     */
    public boolean okToSave(){
    	return plotReady;
    }
	
	public void setMaskButtonEnabled(boolean val){
		maskButton.setEnabled(val);
	}

	public void setCancelButtonEnabled(boolean val){
		cancelButton.setEnabled(val);
	}
	
    @Override
    public void notify(FlipVolumeEvent e) {
        // TODO Auto-generated method stub
         
    }
 
    @Override
    public void notify(LoadedVolumesEvent e) {
        initialize();
    }
 
    @Override
    public void notify(InvertedLvEvent e) {
        // TODO Auto-generated method stub
         
    }

    @Override
    public void notify(Event e) {
        // TODO Auto-generated method stub 
    }
	
	/*When a filter is removed refresh the plot*/
	public void notify(BrainFilterEvent e){
	  plotButtonAction();
	};

	public void notify(SliceFiltersEvent e){};

	public void notify(ViewedLvsEvent e){};

	public void notify(IncorrectLagsSelectedEvent e){};
}