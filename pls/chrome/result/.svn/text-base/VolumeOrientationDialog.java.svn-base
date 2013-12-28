package pls.chrome.result;

import java.awt.Component;
import java.awt.Dimension;
import java.awt.Frame;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.Vector;

import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;

import pls.chrome.result.blvplot.BrainImage;
import pls.chrome.result.blvplot.ColorGradient;
import pls.chrome.result.controller.ResultsCommandManager;
import pls.chrome.result.controller.observer.Event;
import pls.chrome.result.controller.observer.Observer;
import pls.chrome.result.controller.observer.datachange.FlipVolumeEvent;
import pls.chrome.result.controller.observer.selection.SelectedDataTypeChangedEvent;
import pls.chrome.result.controller.observer.selection.SelectedLvChangedEvent;
import pls.chrome.result.controller.observer.selection.SelectedVolumeChangedEvent;
import pls.chrome.result.controller.observer.selection.SelectionEvent;
import pls.chrome.result.controller.observer.selection.SelectionObserver;
import pls.chrome.result.model.BrainData;
import pls.chrome.result.model.GeneralRepository;
import pls.chrome.result.model.ResultModel;

@SuppressWarnings("serial")
public class VolumeOrientationDialog extends JDialog implements ActionListener,
SelectionObserver {
	private static VolumeOrientationDialog dialog;
	
	private JButton mOkButton = new JButton("Ok");
	private JButton mCancelButton = new JButton("Cancel");
	
	private JComboBox mFilesComboBox = new JComboBox();
	
	private PreviewBrainImage mBrainImage;
	
	private int mSliceNum = 0;
	private ArrayList<Integer> mSlice;
	
	JComboBox mViewBox;
	
	private boolean mFlipHorizontal = false;
	private boolean mFlipVertical = false;
	
	private GeneralRepository mRepository;
	
	private VolumeOrientationDialog(Frame owner, GeneralRepository repository) {
		super(owner, "Volume Orientation Dialog");
		
		mRepository = repository;
		
		mRepository.getPublisher().registerObserver(this);
		
		setLayout(new BoxLayout(getContentPane(), BoxLayout.Y_AXIS));
	
		for (String key : repository.getModels() ) {
			mFilesComboBox.addItem(key);
		}
		
		add(new JLabel("Select file"));
		add(mFilesComboBox);
		mFilesComboBox.setMaximumSize(new Dimension(mFilesComboBox.getMaximumSize().width, 25));
		
		add(new JLabel("Select View"));
		mViewBox = new JComboBox();
		mViewBox.setMaximumSize(new Dimension(mViewBox.getMaximumSize().width, 25));
		mViewBox.addItem("Axial");
		mViewBox.addItem("Sagittal");
		mViewBox.addItem("Coronal");
		
		add(mViewBox);
		
		JPanel flipButtonsPanel = new JPanel();
		JButton flipHorizontal = new JButton("Flip Horizontal");
		flipHorizontal.setActionCommand("flip horizontal");
		flipHorizontal.addActionListener(this);
		flipButtonsPanel.add(flipHorizontal);
		JButton flipVertical = new JButton("Flip Vertical");
		flipVertical.setActionCommand("flip vertical");
		flipVertical.addActionListener(this);
		flipButtonsPanel.add(flipVertical);
		add(flipButtonsPanel);
		
		int brainView = mRepository.getImagePropertiesModel().getBrainView();
		mViewBox.setSelectedIndex(brainView);
		
		ResultModel model = mRepository.getGeneral();
		BrainData bData = model.getBrainData();
		
		int[] coords = model.getSelectionModel().getSelectedVoxel();
		int lag = coords[3];
		int coord1d = bData.convert3Dto1D(coords[0], coords[1], coords[2]);
		coords = bData.convert1DtoView(coord1d, brainView);
		
		double[] colours = bData.getMaxMinThresh();
		ColorGradient colGrad = new ColorGradient(colours[0], colours[1], colours[2]);
		
		mSliceNum = coords[2] - 1;
		
		mSlice = (ArrayList<Integer>)bData.getSlices(brainView).get(mSliceNum).clone();
		
		mBrainImage = new PreviewBrainImage(model, mSlice,
				mRepository.getSelectedResultFile(), model.getSelectedDataType(), bData.getLv(),
				lag, mSliceNum, brainView, colGrad);
		
		mBrainImage.setAlignmentX(BrainImage.CENTER_ALIGNMENT);
		
		
		JList historyList = new JList(new Vector<String>(mRepository.getGeneral().getMirrorHistory()));
		JScrollPane scroller = new JScrollPane(historyList);
		scroller.setPreferredSize(new Dimension(200, 200));
		
		JPanel panel = new JPanel();
//		panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
		JPanel imagePanel = new JPanel();
		imagePanel.setLayout(new BoxLayout(imagePanel, BoxLayout.Y_AXIS));
		imagePanel.add(new JLabel("Preview Image:"));
		imagePanel.add(mBrainImage);
		JPanel historyPanel = new JPanel();
		historyPanel.setLayout(new BoxLayout(historyPanel, BoxLayout.Y_AXIS));
		historyPanel.add(new JLabel("Mirror History:"));
		historyPanel.add(scroller);
//		historyPanel.add(historyList);
		
		panel.add(imagePanel);
		panel.add(historyPanel);
		
		add(panel);
		
		JPanel buttonsPanel = new JPanel();
		buttonsPanel.add(mOkButton);
		buttonsPanel.add(mCancelButton);
		
		add(buttonsPanel);
		
		mViewBox.addActionListener(this);
		mOkButton.addActionListener(this);
		mCancelButton.addActionListener(this);
		mFilesComboBox.addActionListener(this);
	}
	
	private void refreshImage() {
		ResultModel model = mRepository.getGeneral(mFilesComboBox.getSelectedItem().toString());
		BrainData bData = model.getBrainData();
		
		double[] colours = bData.getMaxMinThresh();
		ColorGradient colGrad = new ColorGradient(colours[0], colours[1], colours[2]);
		
		int[] coords = model.getSelectionModel().getSelectedVoxel();
		
		mBrainImage.flipHorizontal = mFlipHorizontal;
		mBrainImage.flipVertical = mFlipVertical;
		mBrainImage.mResultModel = model;
		mBrainImage.type = model.getSelectedDataType();
		mBrainImage.lvNum = bData.getLv();
		mBrainImage.lagNum = coords[3];
		mBrainImage.brainView = mViewBox.getSelectedIndex();
		mBrainImage.colGrad = colGrad;
		mBrainImage.sliceNum = mSliceNum;
		mBrainImage.mCoordinates = mSlice;
		
		mBrainImage.reInitializeImage();
		
		repaint();
	}
	
    public static void showDialog(Component frameComp, GeneralRepository repository) {
        Frame frame = JOptionPane.getFrameForComponent(frameComp);
        
        dialog = new VolumeOrientationDialog(frame, repository);
        dialog.setSize(640, 480);
        dialog.setResizable(false);
        dialog.setVisible(true);
    }
    
    public void updateFile() {
    	mFilesComboBox.removeActionListener(this);
    	
    	mFilesComboBox.setSelectedItem(mRepository.getSelectedResultFile() );
    	
    	mFilesComboBox.addActionListener(this);
    	
    	refreshImage();
    }

	public void actionPerformed(ActionEvent e) {
		if (e.getActionCommand().equals("flip horizontal") ) {
			flipCoordsHorizontal();
			
			mFlipHorizontal = !mFlipHorizontal;
			
			refreshImage();
		}
		else if (e.getActionCommand().equals("flip vertical") ) {
			flipCoordsVertical();
			
			mFlipVertical = !mFlipVertical;
			
			refreshImage();
		}
		else if (e.getSource() == mOkButton) {
			int brainView = mViewBox.getSelectedIndex();
			
//			if (mFlipHorizontal) {
//				mRepository.getGeneral().flipHorizontal(brainView);
//			}
//			if (mFlipVertical) {
//				mRepository.getGeneral().flipVertical(brainView);
//			}
//			
//			mRepository.getPublisher().publishEvent(new FlipVolumeEvent() );
			
			ResultsCommandManager.flipVolume(brainView, mFlipHorizontal, mFlipVertical);
			
			dispose();
		}
		else if (e.getSource() == mCancelButton) {
			dispose();
		}
		else if (e.getSource() == mViewBox) {
			mFlipHorizontal = false;
			mFlipVertical = false;
			
			int brainView = mViewBox.getSelectedIndex();
			ResultModel model = mRepository.getGeneral();
			BrainData bData = model.getBrainData();
			
			int[] coords = model.getSelectionModel().getSelectedVoxel();
			int coord1d = bData.convert3Dto1D(coords[0], coords[1], coords[2]);
			coords = bData.convert1DtoView(coord1d, brainView);
			
			mSliceNum = coords[2] - 1;
			mSlice = (ArrayList<Integer>)bData.getSlices(brainView).get(mSliceNum).clone();
			refreshImage();
		}
		else if (e.getSource() == mFilesComboBox) {
			ResultsCommandManager.selectResultFile(mFilesComboBox.getSelectedItem().toString() );
		}
	}
	
	private void flipCoordsHorizontal() {
		int brainView = mViewBox.getSelectedIndex();
		BrainData bData = mRepository.getGeneral().getBrainData();
		int width = bData.getWidth(brainView);
		
		for (int i = 0; i < mSlice.size(); ++i) {
			int coord1D = mSlice.get(i);
			int[] coordView = bData.convert1DtoView(coord1D, brainView);
			int newX = width - coordView[0];
			
			coord1D = bData.convertViewto1D(newX, coordView[1], coordView[2], brainView);
			mSlice.set(i, coord1D);
		}
	}
	
	private void flipCoordsVertical() {
		int brainView = mViewBox.getSelectedIndex();
		BrainData bData = mRepository.getGeneral().getBrainData();
		int height = bData.getHeight(brainView);
		
		for (int i = 0; i < mSlice.size(); ++i) {
			int coord1D = mSlice.get(i);
			int[] coordView = bData.convert1DtoView(coord1D, brainView);
			int newY = height - coordView[1];
			
			coord1D = bData.convertViewto1D(coordView[0], newY, coordView[2], brainView);
			mSlice.set(i, coord1D);
		}
	}
	
	private void updateSelection() {
		int brainView = mViewBox.getSelectedIndex();
		
		ResultModel model = mRepository.getGeneral();
		BrainData bData = model.getBrainData();
		
		int[] coords = model.getSelectionModel().getSelectedVoxel();
		int lag = coords[3];
		
		mBrainImage.lagNum = lag;
		
		int coord1d = bData.convert3Dto1D(coords[0], coords[1], coords[2]);
		coords = bData.convert1DtoView(coord1d, brainView);
		
		mSliceNum = coords[2] - 1;
		
		mSlice = (ArrayList<Integer>)bData.getSlices(brainView).get(mSliceNum).clone();
		
		if (mFlipHorizontal) {
			flipCoordsHorizontal();
		}
		if (mFlipVertical) {
			flipCoordsVertical();
		}
		
		refreshImage();
	}

	@Override
	public void notify(SelectedDataTypeChangedEvent e) {
		refreshImage();
	}

	@Override
	public void notify(SelectedLvChangedEvent e) {
		refreshImage();
	}

	@Override
	public void notify(SelectedVolumeChangedEvent e) {
		updateFile();
	}

	@Override
	public void notify(SelectionEvent e) {
		updateSelection();
	}

	@Override
	public void notify(Event e) {}
}
