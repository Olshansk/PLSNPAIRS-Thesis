package pls.chrome.result;

import java.awt.Component;
import java.awt.Dimension;
import java.awt.Frame;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.util.ArrayList;

import javax.swing.BoxLayout;
import javax.swing.DefaultListModel;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextField;

import pls.chrome.result.blvplot.ColorBarPanel;
import pls.chrome.result.controller.ResultsCommandManager;
import pls.chrome.result.controller.observer.Event;
import pls.chrome.result.controller.observer.datachange.DataChangeObserver;
import pls.chrome.result.controller.observer.datachange.FlipVolumeEvent;
import pls.chrome.result.controller.observer.datachange.InvertedLvEvent;
import pls.chrome.result.controller.observer.datachange.LoadedVolumesEvent;
import pls.chrome.result.model.GeneralRepository;
import pls.shared.AnalyzeImageFileFilter;
import pls.shared.BfMRIResultFileFilter;
import pls.shared.NiftiImageFileFilter;
import pls.shared.NpairsfMRIResultFileFilter;
import pls.shared.PetResultFileFilter;
import pls.shared.ResultFileFilter;
import pls.shared.fMRIResultFileFilter;


@SuppressWarnings("serial")
public class LoadedVolumesDialog extends JDialog implements ActionListener,
DataChangeObserver {
	private static LoadedVolumesDialog dialog;
	private static String lastPath = ".";

	private JButton mBrowseButton = new JButton("Browse");
	private JButton mAddButton = new JButton("Add");
	private JButton mDeleteButton = new JButton("Delete");
	private JButton mOkButton = new JButton("Ok");
	private JButton mCancelButton = new JButton("Cancel");
	
	private JTextField mFilenameField = new JTextField(15);
	
	DefaultListModel mListModel = new DefaultListModel();
	JList mFileList = new JList(mListModel);

	private GeneralRepository mRepository;

	private boolean mDisposeMe = false;

	private LoadedVolumesDialog(Frame owner, GeneralRepository repository) {
		super(owner, "Volumes Dialog");
		
		mRepository = repository;
		mRepository.getPublisher().registerObserver(this);
		
		setLayout(new BoxLayout(getContentPane(), BoxLayout.Y_AXIS));
		
		add(new JLabel("Volumes:"));
		
		for (String key : repository.getModels() ) {
			mListModel.addElement(key);
		}
		
		JScrollPane filesScrollPane = new JScrollPane(mFileList);
		add(filesScrollPane);
		
		JPanel newFilePanel = new JPanel();
		newFilePanel.setLayout(new BoxLayout(newFilePanel, BoxLayout.X_AXIS));
		newFilePanel.add(mFilenameField);
		newFilePanel.add(mBrowseButton);

		
		add(newFilePanel);
		
		JPanel addDeletePanel = new JPanel();
		addDeletePanel.setLayout(new BoxLayout(addDeletePanel, BoxLayout.X_AXIS));
		addDeletePanel.add(mAddButton);
		addDeletePanel.add(mDeleteButton);
		
		add(addDeletePanel);
		
		JPanel okCancelPanel = new JPanel();
		okCancelPanel.setLayout(new BoxLayout(okCancelPanel, BoxLayout.X_AXIS));
		okCancelPanel.add(mOkButton);
		okCancelPanel.add(mCancelButton);
		
		add(okCancelPanel);
		
		mBrowseButton.addActionListener(this);
		mAddButton.addActionListener(this);
		mDeleteButton.addActionListener(this);
		mOkButton.addActionListener(this);
		mCancelButton.addActionListener(this);
		
		filesScrollPane.setPreferredSize(new Dimension(620, 460 - okCancelPanel.getPreferredSize().height) );
	}

	/**
	 * Sets the parent directory to open next time the user wants load result
	 * files.
	 * @param newPath The parent directory to search in.
	 */
	public static void setLastPath(String newPath){
		lastPath = newPath;
	}

    public static void showDialog(Component frameComp, GeneralRepository repository) {
        Frame frame = JOptionPane.getFrameForComponent(frameComp);
        
        dialog = new LoadedVolumesDialog(frame, repository);
        dialog.setSize(640, 480);
        dialog.setResizable(false);
        dialog.setVisible(true);
    }

	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == mBrowseButton) {
			JFileChooser chooser = new JFileChooser(lastPath);
			chooser.setMultiSelectionEnabled(true);
			chooser.addChoosableFileFilter(new NiftiImageFileFilter());
			chooser.addChoosableFileFilter(new AnalyzeImageFileFilter());
			chooser.addChoosableFileFilter(new PetResultFileFilter());
			chooser.addChoosableFileFilter(new NpairsfMRIResultFileFilter());
			chooser.addChoosableFileFilter(new BfMRIResultFileFilter());
			chooser.addChoosableFileFilter(new fMRIResultFileFilter());
			chooser.setFileFilter(new ResultFileFilter());

			chooser.setPreferredSize(new Dimension(680,480));
			int option = chooser.showOpenDialog(this);
			if (option == JFileChooser.APPROVE_OPTION) {
				File[] selectedFiles = chooser.getSelectedFiles();

				if(selectedFiles.length > 0)
					setLastPath(PlsResultLoader.getPrefix(
							selectedFiles[0].getAbsolutePath()));

				String concatenatedFiles = "";
				for (File f : selectedFiles) {
					concatenatedFiles += f.getAbsolutePath() + ";";
				}
				
				mFilenameField.setText(concatenatedFiles);
			}
		}
		else if (e.getSource() == mAddButton) {
			String concatenatedFiles = mFilenameField.getText();
			String[] filenames = concatenatedFiles.split(";");
			
			for(String filename : filenames) {
				if (!filename.equals("")) {
					File file = new File(filename);
					
					if (file.exists() ) {
						mListModel.addElement(filename);
					}
					else {
						JOptionPane.showMessageDialog(null, 
								"The file " + filename + " does not exist.");
					}
				}
			}
			
			mFilenameField.setText("");
			mDeleteButton.setEnabled(true);
			mFileList.repaint();
		}
		else if (e.getSource() == mDeleteButton) {
			Object[] objectsToRemove = mFileList.getSelectedValues();
			
			for (Object o : objectsToRemove) {
				mListModel.removeElement(o);
				
				if (mListModel.getSize() < 1) {
					mDeleteButton.setEnabled(false);
				}
			}
			
			mFileList.repaint(); 
			
			ColorBarPanel cb = GeneralRepository.getColorBarPanel();
			if (cb != null) cb.refreshPanel();
		}
		else if (e.getSource() == mOkButton) {
			int numItems = mListModel.getSize();
			
			// There should be at least one result file
			if (numItems > 0 ) {
				ArrayList<String> volumes = new ArrayList<String>(numItems);
				
				for (int i = 0; i < numItems; ++i) {
					volumes.add((String)mListModel.get(i));
				}
				
				boolean tryAgain = true;
				
				while (tryAgain) {
					try {
						ResultsCommandManager.loadFiles(volumes);
						tryAgain = false;
					} catch (NullPointerException q) { } //do nada
				}
				
				ResultsCommandManager.toggleInverted();
				ResultsCommandManager.toggleInverted();
				/*ColorBarPanel cb = GeneralRepository.getColorBarPanel();
				if (cb != null) cb.refreshPanel();
				mFileList.repaint();*/
				
				//TODO Something is wrong with the way the colorscale is being
				//calculated when files are added. This fixes it, but its still
				//just a kludge.
				mRepository.calculateColourScale();

				if (mDisposeMe) {
					this.dispose();
				}
				else {
					JOptionPane.showMessageDialog(this, 
							"One or more files didn't load properly. " +
							"Please try again.");
				}
			}
			else {
				JOptionPane.showMessageDialog(this, "The list of result " +
						"files is empty.  Please include at least one " +
						"result file.");
			}
		}
		else if (e.getSource() == mCancelButton) {
			this.dispose();
		}
	}
	
	public void dispose() {
		mRepository.getPublisher().unregisterObserver(this);
		
		super.dispose();
	}
	
	@Override
	public void notify(Event e) {}

	@Override
	public void notify(FlipVolumeEvent e) {}

	@Override
	public void notify(LoadedVolumesEvent e) {
		// If we receive this event, the command was successful.
		// So tell the frame to dispose itself after the command has finished.
		mDisposeMe  = true;
	}
	
	@Override
	public void notify(InvertedLvEvent e) {}
}
