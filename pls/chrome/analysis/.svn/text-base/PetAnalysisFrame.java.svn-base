package pls.chrome.analysis;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.FocusEvent;
import java.awt.event.FocusListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;

import javax.swing.*;
import javax.swing.table.*;
import javax.swing.border.Border;

//import npairs.NpairsSetupParameters;

import pls.chrome.shared.BaseSaveMenuBar;
import pls.chrome.shared.ContrastsFrame;
import pls.chrome.shared.ProgressDialog;
import pls.shared.PetSessionFileFilter;
import pls.shared.PetResultFileFilter;
import pls.shared.PetContrastFileFilter;

import pls.shared.PetDatamatFileFilter;

import java.io.File;
import java.util.Vector;

import pls.analysis.Analysis;


@SuppressWarnings("serial")
public class PetAnalysisFrame extends JFrame {
	
	public PetAnalysisFrame(final boolean isBlockedFmri) {
		super("PLS Analysis");
	    setDefaultCloseOperation(DISPOSE_ON_CLOSE);
	    final JFrame frame = this;
        
        final PetDatamatProfilesPanel sessionProfilePanel = new PetDatamatProfilesPanel();
        final PetPermBootPanel permBootPanel = new PetPermBootPanel();
        final PetPlsOptionPanel plsOptionPanel = new PetPlsOptionPanel();
        final PetNpairsOptionsPanel npairsOptionPanel = new PetNpairsOptionsPanel();
        
	    setJMenuBar(new PetAnalysisMenuBar(frame, sessionProfilePanel.conditionSelection, sessionProfilePanel.behaviorBlockConditionSelection, sessionProfilePanel.sessionProfiles));
        
        JPanel containerPane = new JPanel();
        containerPane.setLayout(new BoxLayout(containerPane, BoxLayout.Y_AXIS));
        containerPane.add(sessionProfilePanel);
        containerPane.add(plsOptionPanel);
        containerPane.add(permBootPanel);
        containerPane.add(npairsOptionPanel);
	    
	    final JTextField resultsFilenameField = new JTextField();
	    resultsFilenameField.setColumns(12);
		JButton resultsButton = new JButton("Results File Name", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Import16.gif")));
		resultsButton.setIconTextGap(15);
	    
	    resultsButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				JFileChooser chooser = new JFileChooser(".");
			    chooser.setFileFilter(new PetResultFileFilter());
				int option = chooser.showDialog(PetAnalysisFrame.this, "Save results to");
				if(option == JFileChooser.APPROVE_OPTION) {
					if(!chooser.getSelectedFile().getName().contains(".")) {
						resultsFilenameField.setText(chooser.getSelectedFile().getAbsolutePath() + "_PETresult.mat");
					} else {
						resultsFilenameField.setText(chooser.getSelectedFile().getAbsolutePath());
					}
				}
			}
		});
	    JPanel resultsPane = new JPanel(new GridLayout(1, 0, 10, 0));
	    resultsPane.add(resultsFilenameField);
	    resultsPane.add(resultsButton);
	    resultsPane.setBorder(BorderFactory.createEmptyBorder(5,0,5,0));
        
		final JButton runButton = new JButton("Run", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/media/Play16.gif")));
		runButton.setIconTextGap(15);
		runButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				System.out.println("run buton petrun cagriliyor");
				new PetRunPls(frame, 2, sessionProfilePanel, permBootPanel, plsOptionPanel, npairsOptionPanel, resultsFilenameField);
			}
		});
		
        JPanel mainPane = new JPanel(new BorderLayout());
		mainPane.add(containerPane, BorderLayout.NORTH);
		mainPane.add(resultsPane, BorderLayout.CENTER);
		mainPane.add(runButton, BorderLayout.SOUTH);
		mainPane.setBorder(BorderFactory.createEmptyBorder(10,10,10,10));
		
        add(mainPane);
	    
        // Display the window
        pack();
        
        // Position the frame on the screen
        Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
        int x = (int)(screen.getWidth() - getWidth()) / 2;
        int y = (int)(screen.getHeight() - getHeight()) / 2;
        setLocation(x, y);
        
        setResizable(false);
        
        setVisible(true);
	}
}

final class PetRunPls {
	
	public PetRunPls(JFrame parent, int imagingType, PetDatamatProfilesPanel datamatProfilePanel, PetPermBootPanel permBootPanel, PetPlsOptionPanel plsOptionPanel, PetNpairsOptionsPanel npairsOptionPanel, JTextField resultsFilenameField) {
		// Gather input from gui and validate
		Vector<String[]> datamatProfiles = datamatProfilePanel.sessionProfiles;

		if(plsOptionPanel.radioButtons[1].isSelected() || plsOptionPanel.radioButtons[3].isSelected()) {
		   
			/************/
			//write a code to check that number of subject is at least 3 
			/************/
			/*	
			int count = 0;
			for(String[] sp : datamatProfiles) {
				count += sp.length;
			}
			if(count < 3) {
				JOptionPane.showMessageDialog(null, "Behavior and Multiblock analyses require at least 3 subjects.\nAlso, their behavior data values cannot all be the same for any group.", "Error", JOptionPane.ERROR_MESSAGE);
				return;
			}
			*/
		}
		
		int[] conditionSelection = new int[datamatProfilePanel.conditionSelection.size()];
		for(int i = 0; i < conditionSelection.length; i++) {
			conditionSelection[i] = datamatProfilePanel.conditionSelection.get(i).intValue();
		}
		
		int[] behaviorBlockConditionSelection = new int[datamatProfilePanel.behaviorBlockConditionSelection.size()];
		for(int i = 0; i < behaviorBlockConditionSelection.length; i++) {
			behaviorBlockConditionSelection[i] = datamatProfilePanel.behaviorBlockConditionSelection.get(i).intValue();
		}
		
		String behaviorFilename = plsOptionPanel.behaviorFilenameField.getText().trim();
		String contrastFilename = plsOptionPanel.contrastFilenameField.getText().trim();
		
		String resultsFilename = resultsFilenameField.getText().trim();
		
		if(resultsFilename.equals("")) {
			JOptionPane.showMessageDialog(null, "Please enter a results filename.", "Error", JOptionPane.ERROR_MESSAGE);
			return;
		}
		
		String posthocFilename = plsOptionPanel.posthocFilenameField.getText();
		int numPermutations = Integer.parseInt(permBootPanel.numPermsField.getText());
		int numBootstraps = Integer.parseInt(permBootPanel.numBootsField.getText());
		double confidenceLevel = Double.parseDouble(permBootPanel.confidenceField.getText());
		
		/******/
		//decide on that check shoul it be here, number of the subject check must be done after getting the number of the subjecets for each datamat
		/******/
		/*
		for(int i = 0; i < datamatProfiles.size(); i++) {
			if(numBootstraps > 0 && datamatProfiles.get(i).length < 3) {
				JOptionPane.showMessageDialog(null, "Bootstrap requires 3 or more subjects per group.", "Too few subjects in a group", JOptionPane.ERROR_MESSAGE);
				return;
			}
		}
		*/
		int plsType = 0;
		for(int i = 0; i < plsOptionPanel.radioButtons.length; i++) {
			if(plsOptionPanel.radioButtons[i].isSelected()) {
				plsType = i;
			}
		}
		
		// Gather NPAIRS data into one variable
//		NpairsSetupParameters npairsSetup = new NpairsSetupParameters();
//		
//		npairsSetup.reductionfactor = Float.parseFloat(npairsOptionPanel.reductionFactorField.getText());
//		npairsSetup.cvapcset = Integer.parseInt(npairsOptionPanel.numPCsInCVAField.getText());
//		npairsSetup.cvaallpcset = Integer.parseInt(npairsOptionPanel.numPCsInDataField.getText());
//		
//		// Start new analysis
//		Analysis worker = new Analysis(imagingType, plsType, datamatProfiles, contrastFilename, behaviorFilename, posthocFilename, resultsFilename, conditionSelection, behaviorBlockConditionSelection, numPermutations, numBootstraps, confidenceLevel, npairsSetup);
//		new ProgressDialog(parent, numPermutations + numBootstraps + 3, worker);
	}
}

@SuppressWarnings("serial")
final class PetDatamatProfilesPanel extends JPanel {
	
	protected Vector<String[]> sessionProfiles = new Vector<String[]>();
	
	protected Vector<Integer> conditionSelection = new Vector<Integer>();
	
	protected Vector<Integer> behaviorBlockConditionSelection = new Vector<Integer>();
	
	public PetDatamatProfilesPanel() {
		
		JPanel mainPanel = new JPanel(new BorderLayout());
		
		final DefaultTableModel model = new DefaultTableModel();
		final JTable jt = new JTable(model){
	        public boolean isCellEditable(int rowIndex, int vColIndex) {
	            return false;
	        }
	    };
		model.addColumn("Group");
		model.addColumn("Datamat Files");
		jt.getColumnModel().getColumn(0).setMaxWidth(50);
		jt.getColumnModel().getColumn(0).setResizable(false);
		jt.getColumnModel().getColumn(1).setResizable(false);
		jt.setPreferredScrollableViewportSize(new Dimension(300, 50));
		
		mainPanel.add(new JScrollPane(jt), BorderLayout.NORTH);
		
		JPanel buttonPanel = new JPanel();
		final JButton addButton = new JButton("Add Group", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Add16.gif")));
		buttonPanel.add(addButton);
		final JButton editButton = new JButton("Edit Group", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Edit16.gif")));
		buttonPanel.add(editButton);
		editButton.setEnabled(false);
		final JButton deleteButton = new JButton("Delete Group", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Delete16.gif")));
		deleteButton.setEnabled(false);
		buttonPanel.add(deleteButton);

		mainPanel.add(buttonPanel, BorderLayout.SOUTH);
		
		jt.addFocusListener(new FocusListener() {
			public void focusLost(FocusEvent ae) {
			}
			public void focusGained(FocusEvent ae) {
				if(jt.getRowCount() > 0) {
					deleteButton.setEnabled(true);
					editButton.setEnabled(true);
				}
			}
		});
		addButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				JFileChooser chooser = new JFileChooser(".");
				chooser.setFileFilter(new PetDatamatFileFilter());
				chooser.setMultiSelectionEnabled(true);
				int option = chooser.showDialog(PetDatamatProfilesPanel.this, "Add files to new group");
				if(option == JFileChooser.APPROVE_OPTION) {
					File[] sf = chooser.getSelectedFiles();
					String filelist = "nothing";
					if(sf.length > 0) {
						filelist = sf[0].getAbsolutePath();
					}
					for (int i = 1; i < sf.length; i++) {
						filelist += "; " + sf[i].getAbsolutePath();
					}
					model.addRow(new Object[]{model.getRowCount() + 1, filelist});
					sessionProfiles.add(filelist.split("\\s*;\\s*"));
					
				}
			}
		});
		editButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				JFileChooser chooser = new JFileChooser(".");
			    chooser.setFileFilter(new PetDatamatFileFilter());
				chooser.setMultiSelectionEnabled(true);
				for(int r : jt.getSelectedRows()) {
					int option = chooser.showDialog(PetDatamatProfilesPanel.this, "Select new file(s) for group " + (r + 1));
					if (option == JFileChooser.APPROVE_OPTION) {
						File[] sf = chooser.getSelectedFiles();
						String filelist = "nothing";
						if (sf.length > 0) {
							filelist = sf[0].getAbsolutePath();
						}
						for (int i = 1; i < sf.length; i++) {
							filelist += "; " + sf[i].getAbsolutePath();
						}
						jt.setValueAt(filelist, r, 1);
						sessionProfiles.setElementAt(filelist.split("\\s*;\\s*"), r);
					}
				}
			}
		});
		deleteButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				int decr = 0;
				for(int r: jt.getSelectedRows()) {
					model.removeRow(r - decr);
					sessionProfiles.remove(r - decr);
					decr++;
				}
				for(int i = 0; i < jt.getRowCount(); i++) {
					jt.setValueAt(i + 1, i, 0);
				}
				if(jt.getRowCount() == 0) {
					deleteButton.setEnabled(false);
					editButton.setEnabled(false);
					conditionSelection.removeAllElements();
					behaviorBlockConditionSelection.removeAllElements();
				}
			}
		});
		
	    Border border = BorderFactory.createTitledBorder("Datmat files");
	    mainPanel.setBorder(border);
		mainPanel.setPreferredSize(new Dimension(500, 140));
		add(mainPanel);
	}
}


@SuppressWarnings("serial")
final class PetPlsOptionPanel extends JPanel {
	
	protected JRadioButton[] radioButtons = null;
	
	protected JTextField contrastFilenameField  = new JTextField();
	
	protected JTextField behaviorFilenameField  = new JTextField();
	
	protected JTextField posthocFilenameField  = new JTextField();
	
	public PetPlsOptionPanel() {
		JPanel mainPanel = new JPanel(new BorderLayout());
	    ButtonGroup group = new ButtonGroup();
	    radioButtons = new JRadioButton[5];
	
	    radioButtons[0] = new JRadioButton("Mean-Centering PLS");
	    radioButtons[0].setSelected(true);
	    radioButtons[1] = new JRadioButton("Behavior PLS");
	    radioButtons[2] = new JRadioButton("Non-Rotated Task PLS");
	    radioButtons[3] = new JRadioButton("Multiblock PLS");
	    radioButtons[4] = new JRadioButton("NPAIRS with CVA");
	    
	    for(int i = 0; i < radioButtons.length; i++) {
	    	group.add(radioButtons[i]);
	    }
	    
		JPanel radioBox = new JPanel(new GridLayout(2, 2));
	    Border border = BorderFactory.createEtchedBorder();
	    radioBox.setBorder(border);

	    radioBox.add(radioButtons[0]);
	    radioBox.add(radioButtons[1]);
	    radioBox.add(radioButtons[4]);
	    radioBox.add(radioButtons[2]);
	    radioBox.add(radioButtons[3]);
	    
	    contrastFilenameField.setColumns(12);
	    contrastFilenameField.setEnabled(false);
		final JButton loadContrastButton = new JButton("Load Contrast Data", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Open16.gif")));
		loadContrastButton.setIconTextGap(15);
		loadContrastButton.setEnabled(false);
	    
	    loadContrastButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				JFileChooser chooser = new JFileChooser(".");
			    chooser.setFileFilter(new PetContrastFileFilter());
				int option = chooser.showDialog(PetPlsOptionPanel.this, "Select contrast data file");
				if (option == JFileChooser.APPROVE_OPTION) {
					contrastFilenameField.setText(chooser.getSelectedFile().getAbsolutePath());
				}
			}
		});
	    
	    behaviorFilenameField.setColumns(12);
	    behaviorFilenameField.setEnabled(false);
		final JButton behaviorButton = new JButton("Load Behavior Data", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Open16.gif")));
		behaviorButton.setIconTextGap(15);
		behaviorButton.setEnabled(false);
	    
		behaviorButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				JFileChooser chooser = new JFileChooser(".");
				int option = chooser.showDialog(PetPlsOptionPanel.this, "Select behavior data file");
				if (option == JFileChooser.APPROVE_OPTION) {
					behaviorFilenameField.setText(chooser.getSelectedFile().getAbsolutePath());
				}
			}
		});
	    
	    posthocFilenameField.setEnabled(false);
	    posthocFilenameField.setColumns(12);
		final JButton posthocButton = new JButton("Load Posthoc Data", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Open16.gif")));
		posthocButton.setIconTextGap(15);
		posthocButton.setEnabled(false);
	    
		posthocButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				JFileChooser chooser = new JFileChooser(".");
				int option = chooser.showDialog(PetPlsOptionPanel.this, "Select posthoc data file");
				if (option == JFileChooser.APPROVE_OPTION) {
					posthocFilenameField.setText(chooser.getSelectedFile().getAbsolutePath());
				}
			}
		});
				
		ItemListener activateBehaviour = new ItemListener() {
			public void itemStateChanged(ItemEvent ae) {
				boolean opposite = (radioButtons[1].isSelected() || radioButtons[3].isSelected());
				behaviorButton.setEnabled(opposite);
				behaviorFilenameField.setEnabled(opposite);
				posthocFilenameField.setEnabled(opposite);
				posthocButton.setEnabled(opposite);
			}
		};
		
		radioButtons[1].addItemListener(activateBehaviour);
		radioButtons[2].addItemListener(new ItemListener() {
			public void itemStateChanged(ItemEvent ae) {
				boolean opposite = radioButtons[2].isSelected();
				loadContrastButton.setEnabled(opposite);
				contrastFilenameField.setEnabled(opposite);
			}
		});
		radioButtons[3].addItemListener(activateBehaviour);

	    JPanel fieldPane = new JPanel(new GridLayout(0, 1, 10, 10));
	    fieldPane.setBorder(BorderFactory.createEmptyBorder(0,10,0,10));
	    fieldPane.add(contrastFilenameField);
	    fieldPane.add(behaviorFilenameField);
	    fieldPane.add(posthocFilenameField);
	    JPanel buttonPane = new JPanel(new GridLayout(0, 1, 10, 10));
	    buttonPane.setBorder(BorderFactory.createEmptyBorder(0,10,0,10));
	    buttonPane.add(loadContrastButton);
	    buttonPane.add(behaviorButton);
	    buttonPane.add(posthocButton);
	    JPanel topPane = new JPanel(new GridLayout(0,2));
        topPane.add(fieldPane, BorderLayout.CENTER);
        topPane.add(buttonPane, BorderLayout.LINE_END);
        topPane.setBorder(BorderFactory.createEmptyBorder(0,0,10,0));
		
	    Border border2 = BorderFactory.createTitledBorder("PLS Option");
	    mainPanel.setBorder(border2);
		mainPanel.add(radioBox, BorderLayout.NORTH);
		mainPanel.add(topPane, BorderLayout.SOUTH);
		mainPanel.setPreferredSize(new Dimension(500, 190));
		add(mainPanel);
	}
}

@SuppressWarnings("serial")
final class PetPermBootPanel extends JPanel {
	
	protected JTextField numPermsField = null;
	protected JTextField numBootsField = null;
	protected JTextField confidenceField = null;
	
	public PetPermBootPanel() {
		JPanel mainPane = new JPanel(new GridLayout(0, 1));

	    JLabel numPermsLabel = new JLabel("Number of Permutations");
	    numPermsField = new JTextField();
	    numPermsField.setColumns(5);
	    numPermsField.setText("0");
	    numPermsLabel.setLabelFor(numPermsField);

	    JLabel numBootsLabel = new JLabel("<html>Number of Bootstraps <span style=\"color: red;\">(Requires at least 3 subjects per group)</html>");
	    numBootsField = new JTextField();
	    numBootsField.setColumns(5);
	    numBootsField.setText("0");
	    numBootsLabel.setLabelFor(numBootsLabel);

	    JLabel confidenceLabel = new JLabel("Confidence Level");
	    confidenceField = new JTextField();
	    confidenceField.setColumns(5);
	    confidenceField.setText("95");
	    confidenceLabel.setLabelFor(confidenceField);
	    
	    // Lay out the labels in a panel.
	    JPanel labelPane = new JPanel(new GridLayout(0, 1, 0, 5));
        labelPane.add(numPermsLabel);
        labelPane.add(numBootsLabel);
        labelPane.add(confidenceLabel);
        
        // Layout the text fields in a panel.
        JPanel fieldPane = new JPanel(new GridLayout(0, 1, 0, 5));
        fieldPane.add(numPermsField);
        fieldPane.add(numBootsField);
        fieldPane.add(confidenceField);
	    
	    JPanel topPane = new JPanel(new BorderLayout(5, 0));
        topPane.add(fieldPane, BorderLayout.WEST);
        topPane.add(labelPane, BorderLayout.CENTER);
		
	    Border border = BorderFactory.createTitledBorder("Permutations / Bootstraps");
	    mainPane.setBorder(border);
		mainPane.setPreferredSize(new Dimension(500, 90));
	    
	    mainPane.add(topPane);
	    
		add(mainPane);
	}
}

@SuppressWarnings("serial")
final class PetNpairsOptionsPanel extends JPanel {
	
	protected JTextField reductionFactorField = null;
	protected JTextField numPCsInCVAField = null;
	protected JTextField numPCsInDataField = null;
	
	public PetNpairsOptionsPanel() {
		JPanel mainPane = new JPanel(new GridLayout(0, 1));

	    JLabel reductionFactorLabel = new JLabel("Data Reduction Factor");
	    reductionFactorField = new JTextField();
	    reductionFactorField.setColumns(5);
	    reductionFactorField.setText("1.0");
	    reductionFactorLabel.setLabelFor(reductionFactorField);

	    JLabel numPCsInCVALabel = new JLabel("Set of principal components to use in CVA");
	    numPCsInCVAField = new JTextField();
	    numPCsInCVAField.setColumns(5);
	    numPCsInCVAField.setText("0");
	    numPCsInCVALabel.setLabelFor(numPCsInCVALabel);

	    JLabel numPCsInDataLabel = new JLabel("Set of principal components to use in CVA of all data");
	    numPCsInDataField = new JTextField();
	    numPCsInDataField.setColumns(5);
	    numPCsInDataField.setText("0");
	    numPCsInDataLabel.setLabelFor(numPCsInDataField);
	    
	    // Lay out the labels in a panel.
	    JPanel labelPane = new JPanel(new GridLayout(0, 1, 0, 5));
        labelPane.add(reductionFactorLabel);
        labelPane.add(numPCsInCVALabel);
        labelPane.add(numPCsInDataLabel);
        
        // Layout the text fields in a panel.
        JPanel fieldPane = new JPanel(new GridLayout(0, 1, 0, 5));
        fieldPane.add(reductionFactorField);
        fieldPane.add(numPCsInCVAField);
        fieldPane.add(numPCsInDataField);
	    
	    JPanel topPane = new JPanel(new BorderLayout(5, 0));
        topPane.add(fieldPane, BorderLayout.WEST);
        topPane.add(labelPane, BorderLayout.CENTER);
		
	    Border border = BorderFactory.createTitledBorder("NPAIRS options");
	    mainPane.setBorder(border);
		mainPane.setPreferredSize(new Dimension(500, 90));
	    
	    mainPane.add(topPane);
	    
		add(mainPane);
	}
}
@SuppressWarnings("serial")
final class PetAnalysisMenuBar extends BaseSaveMenuBar {
	public PetAnalysisMenuBar(JFrame parent, Vector conditionSelection, Vector behaviorBlockConditionSelection, Vector<String[]> sessionProfiles) {
		super(parent);
		
        // Build the deselect menu
        JMenu deselectMenu = new JMenu("Deselect");
        deselectMenu.setMnemonic('D');
        deselectMenu.getAccessibleContext().setAccessibleDescription("Deselectable variables");
        add(deselectMenu, 1);
        
        JMenuItem deselectConditions = new JMenuItem("Deselect conditions (before loading behavior data)", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Preferences16.gif")));
        deselectConditions.setMnemonic('C');
        deselectConditions.getAccessibleContext().setAccessibleDescription("Filter out conditions you don't want to be a part of the analysis");
        deselectConditions.addActionListener(new PetDeselectConditionsListener(parent, conditionSelection, sessionProfiles));
        deselectMenu.add(deselectConditions);
        
        JMenuItem deselectBehaviourBlockConditions = new JMenuItem("Deselect behavior block conditions for Multiblock PLS (after loading behavior data)", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Preferences16.gif")));
        deselectBehaviourBlockConditions.setMnemonic('B');
        deselectBehaviourBlockConditions.getAccessibleContext().setAccessibleDescription("Filter out conditions you don't want to be a part of the analysis");
        deselectBehaviourBlockConditions.addActionListener(new PetDeselectConditionsListener(parent, behaviorBlockConditionSelection, sessionProfiles));
        deselectMenu.add(deselectBehaviourBlockConditions);
        
        // Build the contrast menu
        JMenu contrastMenu = new JMenu("Contrast");
        contrastMenu.setMnemonic('C');
        contrastMenu.getAccessibleContext().setAccessibleDescription("Contrast");
        add(contrastMenu, 2);
        
        JMenuItem openContrastsWindow = new JMenuItem("Open Contrast Window", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/media/Movie16.gif")));
        openContrastsWindow.setMnemonic('O');
        openContrastsWindow.getAccessibleContext().setAccessibleDescription("Create a contrasts file");
//        openContrastsWindow.addActionListener(new NpairsContrastListener(sessionProfiles));
        contrastMenu.add(openContrastsWindow);
	}
}
final class PetContrastListener implements ActionListener {
	
	private Vector<String[]> sessionProfiles = null;
	
	public PetContrastListener(Vector<String[]> sessionProfiles) {
		this.sessionProfiles = sessionProfiles;
	}
	public void actionPerformed(ActionEvent e) {
		new ContrastsFrame(sessionProfiles);
	}
}
final class PetDeselectConditionsListener implements ActionListener {
	
	private JFrame parent = null;
	
	private Vector conditionSelection = null;
	
	private Vector<String[]> sessionProfiles = null;
	
	public PetDeselectConditionsListener(JFrame parent, Vector conditionSelection, Vector<String[]> sessionProfiles) {
		this.parent = parent;
		this.conditionSelection = conditionSelection;
		this.sessionProfiles = sessionProfiles;
	}
	public void actionPerformed(ActionEvent e) {
		new DeselectConditionsFrame(parent, sessionProfiles, conditionSelection);
	}
}
