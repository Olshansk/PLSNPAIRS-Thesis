package pls.chrome.sessionprofile;

import pls.sessionprofile.PetSubjectInformation;

import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.FlowLayout;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.Vector;

import javax.swing.BorderFactory;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JList;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.JLabel;
import javax.swing.DefaultListModel;
import javax.swing.DefaultListSelectionModel;
import javax.swing.border.Border;
import javax.swing.event.ListSelectionListener;
import javax.swing.event.ListSelectionEvent;


import java.io.File;

import nl.jj.swingx.gui.modal.JModalFrame;

class MergedConditionsClass {
	String conditionName;
	int [] mergedconditionindex;
	public MergedConditionsClass(String cn, int[] in){
		conditionName = cn;
		mergedconditionindex = in;
	}
	/*
	 void display(){
		System.out.println("\n"+conditionName);
		for(int i=0; i<mergedconditionindex.length;i++)
			System.out.print(mergedconditionindex[i]);
	}
	*/
}
public class PetMergeConditionsFrame extends JModalFrame {
	int []selectedconditionindex=null;
	int selectedconditionsize;
	JList conditionlist;
	DefaultListModel model;
	int numCondition;
	String [] con;
	JTextField mergedconditionname;
	Vector <MergedConditionsClass> cond= new Vector<MergedConditionsClass>();
	
	public PetMergeConditionsFrame(PetSessionProfileFrame sessionFrame) {
		super("Merge Conditions");
	    setDefaultCloseOperation(DISPOSE_ON_CLOSE);
		addAdditionalModalToWindow(sessionFrame);
				
		if(sessionFrame.subjectInfo.isEmpty()) {
			JOptionPane.showMessageDialog(null, "You must have a subject to be able to merge conditions.", "Error", JOptionPane.ERROR_MESSAGE);
			dispose();
			return;
		}
		numCondition = sessionFrame.conditionInfo.size();
		selectedconditionsize = 0;				
				
		con = new String[numCondition];
		
		model = new DefaultListModel();
		
		for(int i=0;i<numCondition;i++)
		{
			con[i] = new String(sessionFrame.conditionInfo.get(i)[0]);
			model.add(i,con[i]);		
		}
		
		conditionlist = new JList(model);
		conditionlist.setSelectionMode(DefaultListSelectionModel.MULTIPLE_INTERVAL_SELECTION);
		
		Border border = BorderFactory.createTitledBorder("Conditions");
		JPanel conditionPane = new JPanel(new GridLayout(2, 1, 5, 5));
		conditionPane.setBorder(border);
		conditionPane.add(conditionlist);
	
		JLabel conditionnameLabel = new JLabel("Merged condition name:");
		mergedconditionname = new JTextField();
		mergedconditionname.setColumns(10);
		
		JPanel ex = new JPanel(new FlowLayout());
		ex.add(conditionnameLabel);
		ex.add(mergedconditionname);
		conditionPane.add(ex);

		/*JPanel conPanel = new JPanel(new GridLayout(1, 1, 5, 5));
		conPanel.add(conditionPane);
		*/
		selectedconditionindex= new int[numCondition];
		selectedconditionsize = 0;
		
		JButton mergeButton = new JButton("Merge", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Open16.gif")));
		mergeButton.setIconTextGap(15);
		
		mergeButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae){								
				if(mergedconditionname.getText().equals(""))
				{
		    		JOptionPane.showMessageDialog(null, "You must enter merged condition Name ", "Error", JOptionPane.ERROR_MESSAGE);
				}
				else{
					selectedconditionindex = conditionlist.getSelectedIndices();
					int num= numCondition - selectedconditionindex.length+1;
					String [] newCondList = new String[num];
					
					boolean flag=false;
					int k=0;
					int ni=1;
					model.removeAllElements();
					model.add(0,mergedconditionname.getText());
					newCondList[0] = mergedconditionname.getText();

					//store which conditions are merged as a new condition 					
					MergedConditionsClass mc = new MergedConditionsClass(mergedconditionname.getText(),selectedconditionindex);
					cond.addElement(mc);
					
					for(int j=0;j<con.length;j++)
					{
						flag = false;
						for(int i=0; i<selectedconditionindex.length;i++){
							if(j == selectedconditionindex[i]){
								flag = true;
							}
						}
						if(flag == false){
							String element = con[j];
							newCondList[ni] = con[j];
							model.add(ni,element);
							ni++;							
						}
					
					}
					mergedconditionname.setText(null);
					con = newCondList;
				}
				
			}
		});
		
		JButton doneButton = new JButton("Done", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/Save16.gif")));
		doneButton.setIconTextGap(15);
		
		doneButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae){								
				
			}
		});
		
		JPanel buttonPane = new JPanel(new FlowLayout());
		buttonPane.add(mergeButton);
		buttonPane.add(doneButton);
		
	    JPanel mainPane = new JPanel(new FlowLayout());
	    
	    //mainPane.add(conPanel);	
	    mainPane.add(conditionPane);
	    mainPane.add(buttonPane);
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
	
	public static String getCommonPath(Vector<PetSubjectInformation> subjectInfo) {
		return new File(subjectInfo.get(0).dataDirectory).getParent();
	}
}

final class PetMergeConditionListener implements ActionListener {

	private JFrame parent = null;
	private PetSessionProfileFrame sessionFrame = null;
	
	
	public PetMergeConditionListener(JFrame parent, PetSessionProfileFrame sessionFrame) {
		this.parent = parent;
		this.sessionFrame = sessionFrame;
		
	}
	public void actionPerformed(ActionEvent e) {
		
		parent.dispose();
	}
}

