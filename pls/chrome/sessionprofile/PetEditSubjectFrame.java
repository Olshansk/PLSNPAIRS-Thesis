package pls.chrome.sessionprofile;

import pls.sessionprofile.PetRunInformation;
import pls.sessionprofile.PetSubjectInformation;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.FilenameFilter;
import java.util.Vector;

import javax.swing.BorderFactory;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTabbedPane;
import javax.swing.JTable;
import javax.swing.JTextField;
import javax.swing.border.Border;
import javax.swing.table.DefaultTableModel;

import nl.jj.swingx.gui.modal.JModalFrame;
import pls.chrome.shared.BaseSaveMenuBar;
import pls.shared.AnalyzeNiftiFileFilter;

public class PetEditSubjectFrame extends JModalFrame {
	protected JTextField subjectInitialText = new JTextField("-1");

	public PetEditSubjectFrame(PetSessionProfileFrame sessionFrame) {
		super("Edit Subject");
		
		setDefaultCloseOperation(DISPOSE_ON_CLOSE);
		addAdditionalModalToWindow(sessionFrame);
	    
		JTabbedPane tabs = new JTabbedPane(JTabbedPane.TOP, JTabbedPane.SCROLL_TAB_LAYOUT);
				
		if(sessionFrame.subjectInfo.size() == 0) {
			tabs.add(new PetEditSubjectPanel(sessionFrame, null,0), "Subject 1");
		} else {
			for(int i = 0; i < sessionFrame.subjectInfo.size(); i++) {
				tabs.add(new PetEditSubjectPanel(sessionFrame, sessionFrame.subjectInfo.get(i),i), "Subject " + (i + 1));
			}
		}
	    
		JButton addRunButton = new JButton("Add New Subject", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Add16.gif")));
		
		addRunButton.addActionListener(new PetAddSubjectListener(sessionFrame, tabs));
	    
		JButton removeRunButton = new JButton("Remove Current Subject", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Delete16.gif")));
		removeRunButton.addActionListener(new PetRemoveSubjectListener(tabs,sessionFrame));
		JButton saveRunButton = new JButton("Store And Continue", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/SendMail16.gif")));
		saveRunButton.addActionListener(new PetSubjectSaveListener(tabs, sessionFrame, this));
		
		JPanel buttonPanel = new JPanel();
		buttonPanel.add(addRunButton);
		buttonPanel.add(removeRunButton);
		buttonPanel.add(saveRunButton);
		
		JPanel subjectInitial = new JPanel();
		JLabel subjectInitialLabel = new JLabel("Number of characters for subject initial: ");
		
		subjectInitialText.setColumns(3);
		
		subjectInitial.add(subjectInitialLabel);
		subjectInitial.add(subjectInitialText);
		
		if(sessionFrame.subjectInitial != -1)
			subjectInitialText.setText(sessionFrame.subjectInitial+"");
		
		
		JPanel mainPanel = new JPanel(new BorderLayout());
		mainPanel.add(tabs, BorderLayout.NORTH);
		mainPanel.add(buttonPanel, BorderLayout.SOUTH);
		mainPanel.add(subjectInitial, BorderLayout.CENTER);
		
		add(mainPanel);
				
		setJMenuBar(new BaseSaveMenuBar(this));
		
        pack();
        
        Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
        int x = (int)(screen.getWidth() - getWidth()) / 2;
        int y = (int)(screen.getHeight() - getHeight()) / 2;

        setLocation(x, y);
        
        //setResizable(false);
        setResizable(true);
        setVisible(true);
        
	}
}

final class FileSelectionConditionListener implements ActionListener{
	FileSelectionForCondition parent;
	int numberofcondition;
	String cwd;
	PetEditSubjectPanel peteditsubectPanel; 
	public FileSelectionConditionListener(FileSelectionForCondition parent, int conditionnumber, PetEditSubjectPanel peteditsubectPanel)
	{
		this.peteditsubectPanel = peteditsubectPanel;
		this.parent = parent;
		numberofcondition  = conditionnumber;
			
	}
	public void actionPerformed(ActionEvent e) {
		cwd =  peteditsubectPanel.dataDirectoryField.getText();
		 		
		if(!cwd.equals("")) {				
			JFileChooser chooser = new JFileChooser(cwd);
			chooser.setFileFilter(new AnalyzeNiftiFileFilter());
			chooser.setMultiSelectionEnabled(true);				
			int option = chooser.showDialog(parent, "Select subject File");
			if (option == JFileChooser.APPROVE_OPTION) {
			//parent.subectfile[numberofcondition].setText(chooser.getSelectedFile().getAbsolutePath());					
				parent.subectfile[numberofcondition].setText(chooser.getSelectedFile().getName());
			}
			
		}
	}
}
class FileSelectionForCondition extends JPanel {
	public JTextField [] condition;
	public JTextField [] subectfile;
	public JButton [] selectSubjectFileButton;
	public PetSessionProfileFrame sessionFrame; 
	public PetEditSubjectPanel petEditSubjectPanel;
	public FileSelectionForCondition(PetSessionProfileFrame sessionFrame,PetEditSubjectPanel petEditSubjectPanel, int tabnumber)
	{
		this.petEditSubjectPanel = petEditSubjectPanel;
		this.sessionFrame=sessionFrame;
		int num= sessionFrame.conditionInfo.size();
		condition = new JTextField[num];
		subectfile = new JTextField[num];
		selectSubjectFileButton = new JButton[num];

		for(int i=0; i<num;i++){
			condition[i] = new JTextField();
			condition[i].setText(new String(sessionFrame.conditionInfo.get(i)[0]));
			subectfile[i] = new JTextField();
			selectSubjectFileButton[i] = new JButton("Select Subject File", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Add16.gif")));			
			selectSubjectFileButton[i].addActionListener(new FileSelectionConditionListener(this,i,petEditSubjectPanel));
			setLayout(new GridLayout(num,3,5,5));
			add(condition[i]);
			add(subectfile[i]);
			add(selectSubjectFileButton[i]);			
		}	
		
		//System.out.println("subject info size" +sessionFrame.subjectInfo.size());
		//System.out.println("tanmu,ber" +tabnumber);
		
		if(sessionFrame.subjectInfo.size()>0 && sessionFrame.subjectInfo.size()>tabnumber){
			try{
				for(int i=0; i<num;i++){		
					subectfile[i].setText(new String(sessionFrame.subjectInfo.get(tabnumber).subjectFiles[i]));
				}
			}
			catch(Exception e){}
		}
	}	
}

final class PetEditSubjectPanel extends JPanel {
	
    public JTextField dataDirectoryField = new JTextField();
    
    public JTable table = new JTable();
    
    public FileSelectionForCondition conditionPanle;
    
    public DefaultTableModel model = new DefaultTableModel();
	public static String cwd=".";
    
	public int numCondition;
	
    public PetEditSubjectPanel(PetSessionProfileFrame sessionFrame, PetSubjectInformation subjectInfo, int tabnumber) {
	    dataDirectoryField.setColumns(48);
	    dataDirectoryField.setEditable(false);
	    
	    JLabel dataDirectoryLabel = new JLabel("Data Directory: ");
		JButton selectDataFilesButton = new JButton("Select Data", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Open16.gif")));
		selectDataFilesButton.setIconTextGap(15);
	    
		numCondition = sessionFrame.conditionInfo.size();
		
	    if(subjectInfo != null) {
	    	dataDirectoryField.setText(subjectInfo.dataDirectory);
	    }
	    
	    conditionPanle = new FileSelectionForCondition(sessionFrame, this, tabnumber);
	    
	    
	    table.setModel(model);
	    table.setPreferredScrollableViewportSize(new Dimension(table.getPreferredScrollableViewportSize().width, 160));

	    model.addColumn("Condition Name");
	    model.addColumn("Subject Files");
	    
	    for(int i = 0; i < sessionFrame.conditionInfo.size(); i++) {
	    	String subjectFiles = "";
	    	try {
	    		subjectFiles = subjectInfo.subjectFiles[i];
	    	} catch(Exception ex) {
	    		// No onsets exist
	    		subjectFiles = "0";
	    	}
	    	model.addRow(new String[]{sessionFrame.conditionInfo.get(i)[0], subjectFiles});
	    }
	    	    
	   selectDataFilesButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				if(!dataDirectoryField.getText().equals("")) {
					cwd = dataDirectoryField.getText();
				}
				JFileChooser chooser = new JFileChooser(cwd);
				chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
				int option = chooser.showDialog(PetEditSubjectPanel.this, "Select data files");
				String sf="";
				
				if(option == JFileChooser.APPROVE_OPTION) {
					sf = chooser.getSelectedFile().getAbsolutePath();					
					File dir = new File(sf);
					File[] f = dir.listFiles();
					int k=0;
					for(int i=0; i<f.length && k<numCondition; i++){
						if(f[i].getName().endsWith(".img") || f[i].getName().endsWith(".nii")){
							conditionPanle.subectfile[k].setText(f[i].getName());
							k++;
						}
					}
				}	
				if(PetEditSubjectPanel.cwd.equals("."))
				{
					PetEditSubjectPanel.cwd = sf.substring(0,sf.lastIndexOf("\\"));
					
				}
				dataDirectoryField.setText(sf);	
				
				
			}
		});
	    JPanel directoryPanel = new JPanel(new BorderLayout());
	    directoryPanel.add(dataDirectoryLabel, BorderLayout.WEST);
	    directoryPanel.add(dataDirectoryField, BorderLayout.EAST);
	    JPanel labelPanel = new JPanel(new GridLayout(0, 1, 10, 10));

	    labelPanel.add(directoryPanel);
	    JPanel dataPanel = new JPanel(new GridLayout(2, 0));

	    dataPanel.add(selectDataFilesButton);
	    JPanel tablePanel = new JPanel();
	    
	    Border border = BorderFactory.createTitledBorder("Condition and Files");
	    tablePanel.setBorder(border);
	    	    
	    tablePanel.add(conditionPanle);
        JPanel mainPanel = new JPanel(new BorderLayout(15, 15));
        mainPanel.add(labelPanel, BorderLayout.NORTH);
        mainPanel.add(dataPanel, BorderLayout.CENTER);
        mainPanel.add(tablePanel, BorderLayout.SOUTH);
        add(mainPanel);        
	}
}

final class PetSubjectSaveListener implements ActionListener {
	
	private JTabbedPane tabs = null;
	
	private PetEditSubjectFrame subjectsFrame = null;
	
	private PetSessionProfileFrame sessionFrame = null;
	
	public PetSubjectSaveListener(JTabbedPane tabs, PetSessionProfileFrame sessionFrame, PetEditSubjectFrame subjectsFrame) {
		this.tabs = tabs;
		this.subjectsFrame = subjectsFrame;
		this.sessionFrame = sessionFrame;
	}
	
	public void actionPerformed(ActionEvent e) {
		Vector<PetSubjectInformation> subjectInfo = new Vector<PetSubjectInformation>();		
		
		for(int i = 0; i < tabs.getTabCount(); i++) {
			PetEditSubjectPanel c = ((PetEditSubjectPanel)tabs.getComponentAt(i));
			if(c.table.getCellEditor() != null) {
				c.table.getCellEditor().stopCellEditing();
			}
			String dataDirectory = c.dataDirectoryField.getText().trim();
			
			if(!dataDirectory.equals("")) {				
				int numcondition = sessionFrame.conditionInfo.size();
				String [] subjectFiles= new String[numcondition];
								
				for(int m=0; m<numcondition; m++){
					subjectFiles[m]= new String(c.conditionPanle.subectfile[m].getText());
				}
							
				PetSubjectInformation currSubjectInfo = new PetSubjectInformation(dataDirectory, subjectFiles);
				
				subjectInfo.add(currSubjectInfo);						
				}				
			}			
			sessionFrame.updateSubjects(subjectInfo);				
			subjectsFrame.dispose();	
			sessionFrame.subjectInitial =Integer.parseInt(subjectsFrame.subjectInitialText.getText()); 
		}		  
		
	}


final class PetAddSubjectListener implements ActionListener {
	
	private JTabbedPane tabs = null;
	
	private PetSessionProfileFrame sessionFrame = null;
	
	public PetAddSubjectListener(PetSessionProfileFrame sessionFrame, JTabbedPane tabs) {
		this.tabs = tabs;
		this.sessionFrame = sessionFrame;
	}
	
	public void actionPerformed(ActionEvent e) {
		int subjectIndex = tabs.getTabCount();
		PetEditSubjectPanel c = ((PetEditSubjectPanel)tabs.getComponentAt(subjectIndex-1));
		String cwd = c.dataDirectoryField.getText();
		
		tabs.add(new PetEditSubjectPanel(sessionFrame, null, subjectIndex), "Subject " + (subjectIndex + 1));
		
		tabs.setSelectedIndex(subjectIndex);
	}
}

final class PetRemoveSubjectListener implements ActionListener {
	
	private JTabbedPane tabs = null;
	PetSessionProfileFrame sessionFrame;
	
	public PetRemoveSubjectListener(JTabbedPane tabs,PetSessionProfileFrame sessionFrame) {
		this.tabs = tabs;
		this.sessionFrame = sessionFrame;
	}
	
	public void actionPerformed(ActionEvent e) {
		int subjectIndex = tabs.getSelectedIndex();
		if(subjectIndex < 0) {
			return;
		}
		tabs.remove(subjectIndex);
		for(int i = subjectIndex; i < tabs.getTabCount(); i++) {
			tabs.setTitleAt(i, "Subject " + (i + 1));
		}
		
		//sessionFrame.subjectInfo.removeElementAt(subjectIndex);
		
	}
}